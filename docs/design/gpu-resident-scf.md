# GPU-resident SCF for PM6-D3H4 + COSMO (conventional and MOZYME)

Status: design draft, 2026-11-11. Branch `dev/magma-dsyevd` (MAGMA eigensolver, c755e603).
Target workloads: PM6-D3H4 (and D3H4X) with COSMO, single points and geometry steps, for
protein-ligand systems of ~400–5,000 atoms (2,000–12,500 AOs), with both the conventional SCF
(full diagonalization) and MOZYME (localized orbitals). Running many systems at once is done
outside MOPAC with NVIDIA MPS; nothing here schedules multiple jobs.

## 1. Measurements that drive the design

Whole PTP1B: 4,878 atoms, 12,362 AOs (19 S with d orbitals, 2,444 other heavy atoms, 2,415 H),
COSMO with 8,428 segments, on an RTX PRO 6000 Blackwell server (24 threads on one NUMA node; RTX PRO 6000 Blackwell, 96 GB).
`TIMES` output, wall seconds:

| Step | CPU only | MAGMA first pass (c755e603) | MOZYME (NOFEATHER, CUTOFF=100, 1 core) |
|---|---|---|---|
| HCORE (integrals, once per geometry) | 16.1 | 11.1 | 8.4 |
| Fock build incl. COSMO addfck | 2.6 | 2.5 | ~11.5 together with density, COSMO, and other per-iteration work not split out by `TIMES` |
| Full diagonalization | 40.6–43.7 | 25.0 | — |
| diagg1 / diagg2 (MOZYME annihilation) | — | — | 1.6 / 0.04 |
| Density | 1.9 | 1.8 | (in the 11.5 above) |
| **Per iteration** | **45–49** | **29.8** | **~13.2** |
| Whole SCF | 1,693 s, 46 it. | 1,315 s, 46 it. | ~100 it. at SCFCRT=1e-3 |

Bare MAGMA `dsyevd` (vectors) at N = 12,000 on the same card: 10.0 s (CPU, 16 threads: 36.8 s).

Conclusions:
1. Conventional: the diagonalization dominates (88% on the CPU). The first pass leaves ~15 s
   of the 25 s outside the GPU solve: per-call host allocation and first touch of a 2.4 GB
   workspace, `dtpttr` unpack, MAGMA's host-interface staging (120 CPU-s per call), and two
   1.2 GB PCIe transfers. Device residency removes these.
2. Fock + COSMO is only 2.5 s per iteration today, but once the solve is ~10 s or less it
   becomes a quarter of the iteration. It is memory-bound (4.3 GFLOP against a 3.07 GB
   integral stream) plus COSMO GEMVs over 1.9 GB of `bmat`, a good match for GPU bandwidth.
3. MOZYME: per-iteration cost is not the rotations (0.04 s) but the work around them:
   `rotate` for every near pair every iteration (DIRECT is the default), all-pairs far-field
   loops, the linear-scaling COSMO potential and Fock terms (atoms x segments), density
   from the LMOs, and `tidy`/`check`. All of it is single-threaded: MOPAC has no OpenMP.
4. HCORE is single-threaded on the CPU and its routines keep per-pair scratch in module
   globals (`mndod_C`, `overlaps_C`), so they cannot be threaded as written.

## 2. Principles

- **Device residency.** From the start of an SCF to the end, H, the integrals, P, F, C,
  COSMO matrices, Pulay history and MOZYME LMO arrays live in GPU memory. Per iteration
  only scalars cross PCIe (energy, convergence measures, overflow flags).
- **FP64 throughout.** The study criterion is 1e-6 kcal/mol on heats of formation of order
  1e4 kcal/mol (relative 1e-10). FP32 is a later, opt-in experiment for far-field terms only.
  These cards run FP64 at 1/32 (A4000) to 1/64 (Blackwell) of FP32, so the design favors
  bandwidth-bound and latency-hiding work over FP64 FLOP count.
- **One integral kernel, two modes.** The per-pair integral code (overlap and resonance,
  two-electron, electron-nuclear, core-core) is a device function used to (a) fill stored
  integrals once per geometry, or (b) generate integrals on the fly inside the Fock build
  (direct SCF). The mode is chosen from free VRAM (section 6). The same kernel serves the
  finite-difference gradients (`dcart`/`dhc` evaluate the same per-pair integrals at
  displaced geometries, ~4x HCORE work per gradient).
- **Concurrency inside an iteration** with CUDA streams and events (section 5).
- **CPU path is the reference.** Every kernel is checked against the CPU code. Without
  `-DMOPAC_MAGMA` (and a future `-DMOPAC_CUDA`) the binary is bitwise unchanged.
- **Determinism option.** Reductions into shared blocks use deterministic segmented sums
  by default; FP64 `atomicAdd` versions are allowed behind a switch for speed tests.

## 3. Data layout on the device

Packed lower triangle by rows (= LAPACK 'U' packed) stays the interchange format, so CPU
fallbacks and AUX output keep working. Kernels may keep full square copies where that makes
access coalesced.

Integral array W (conventional): per atom i, the pair blocks (i, j<i) in j order, then the
one-center block. Block size lim_i x lim_j with lim = 1 (H), 10 (sp), 45 (spd). The offset
has a closed form with an exclusive scan: L(k) = sum_{l<k} lim_l,
base(i) = sum_{k<i} lim_k (L(k) + lim_k), offset(i,j) = base(i) + lim_i L(j). So every pair
can be computed independently (the CPU code uses a running cursor). n2elec is checked against
2^31 today; the device version uses 64-bit offsets.

MOZYME: block-CSR by atom row over the NDDO pairs inside CUTOF2 (`nijbo`, or its CSR form),
LMO coefficient pools with per-LMO capacity (ELL-style slot or chunked pool with an atomic
allocator) so that fill-in during rotations does not need the global `tidy` reshuffle.

Whole-PTP1B footprints (FP64):

| Item | Size |
|---|---|
| W, all pairs (conventional) | 3.07 GB |
| Packed H, F, P, PA (each) | 0.61 GB |
| C (square) | 1.22 GB |
| Eigensolver: full F + workspace | ~3.7 GB |
| Pulay history (npulay = 10, pold + pold2) | 12.2 GB (host today) |
| Pulay work arrays (3 x N^2) | 3.7 GB |
| COSMO bmat (lm61 = 27,710 x nps 8,428) + amat (packed nps^2/2) | 1.87 + 0.28 GB |
| MOZYME total (LMO pools, sparse P/F/H, fmo, nijbo, COSMO) | ~0.5–0.8 GB |

## 4. Kernels

### 4.1 Integrals (HCORE), one block (or warp) per atom pair
- Overlap `diat`/`diat2`/`ss` (Slater-type, analytic; `set`/`aintgs`/`bintgs` auxiliaries),
  rotated to the molecular frame; resonance beta*S written directly into H(i,j)
  (pair-exclusive block, plain stores).
- Two-electron `rotatd`: `rotmat`, `reppd` (22 local sp integrals), `spcore`, `reppd2`
  (d terms), `tx` + `met` rotation, specialized per pair class (H-H, sp-H, sp-sp, d-x) so
  H-H pairs never touch the 2025-element scratch. Per-thread/block local state replaces
  the module globals.
- Electron-nuclear e1b/e2a into the diagonal blocks of i and j (shared: segmented reduction
  per atom), core-core `ccrep` summed into enuclr. The `ccrep` write into `alpb` becomes a
  precomputed pair-parameter table. The duplicate `elenuc` call is dropped.
- One-center blocks (`wstore`) per atom.
- Feathering (`l_feather`) and NOFEATHER honored identically to the CPU.

### 4.2 Conventional Fock build (per iteration)
- Exchange: one block per atom pair; reads the off-diagonal P(i,j) block and W(i,j),
  writes F(i,j). No conflicts.
- Coulomb: one block per atom A looping over partners B, accumulating W_AB P_BB into F_AA
  in registers (both index orders of the stored W). No atomics.
- One-center `fock1`: one block per atom.
- Direct mode: the integral device function is called inside the exchange/Coulomb kernels
  for the pair being contracted, so W for that pair lives only in registers/shared memory.

### 4.3 COSMO (conventional `addfck`, per Fock build)
- phi = bmat^T q, then the two packed triangular solves with the Cholesky factor of A,
  then F(ipiden) -= fcon * bmat * qsc: cuBLAS dgemv and dtpsv (or one dense dpotrs on a
  full factor), with the gather/scatter through `ipiden`/`gden` as small kernels. The CPU
  version uses a stride-lm61 GEMV over a 1.9 GB matrix and hand-written serial solves.
- `coscl1` (Cholesky of A once per geometry): MAGMA/cuSOLVER dpotrf on the device.

### 4.4 Energy, Pulay, diagonalization, density
- helect: reduction over the packed arrays.
- Pulay: device-resident history (packed), commutator via cuBLAS dsymm on full copies,
  dot products and the small DIIS solve on the device; extrapolated F written in place.
- Diagonalization: MAGMA `magma_dsyevd_gpu` (device interface) on one GPU, or the
  multi-GPU `magma_dsyevd_m` path when the matrix sits in host-pinned memory is the
  multi-GPU fallback. Workspaces allocated once per SCF and reused.
- Density: cuBLAS dsyrk on C (occupied columns), then pack to P on the device.

### 4.5 MOZYME (per iteration)
- Sparse Fock `fz2n`: one block per NDDO pair (Coulomb/exchange as 4.2 on block-CSR),
  integrals stored (near pairs: ~175 MB) or direct.
- Far field: all-pairs monopole/dipole as an N-body kernel (11.9 M pairs).
- `diagg1`: one warp/block per virtual LMO; stage its atoms and coefficients in shared
  memory; candidate occupied LMOs from atom neighbor lists and an atom-to-LMO inverted
  index instead of the O(nocc x nvir) rebuild test; compact (i, j, F_ij) output.
- `diagg2`: rotations conflict when they share an LMO. Edge-color the bipartite
  occupied/virtual interaction graph; each color class is a matching applied in parallel
  as exact 2x2 rotations (orthonormality preserved). Results match the CPU to SCF
  tolerance, not bitwise. Fill-in uses per-LMO capacity with an overflow flag.
- Density from LMOs: one block per occupied LMO (outer products into pair blocks with
  segmented reductions or atomics), or a gather per pair over an inverted index.
- `helecz`, `eimp`, `cnvgz`, `check`, `tidy`: flat reductions and per-LMO kernels.
- Linear-scaling COSMO `addfckz`: atoms x segments potential and Fock kernels (N-body),
  CG solve on the device (A*q direct O(nps^2) for nps up to ~2e4, otherwise port the FMM),
  block-Jacobi preconditioner via batched dpotrf/dpotri once per surface.

### 4.6 D3H4 (geometry only)
- D3: coordination numbers and pair dispersion as N-body kernels; gradients likewise.
- H4: H-bond search and terms per H-bond; replace the finite-difference H4 gradient with
  an analytic one later.
- Runs on its own stream concurrently with the SCF (it depends only on coordinates).

## 5. Concurrency and streaming

Within one conventional iteration (dependencies: P -> {Fock 2e, COSMO, one-center} -> F ->
{energy, Pulay} -> diag -> C -> density -> P):

- Stream A: two-electron Fock (exchange + Coulomb). Stream B: COSMO potential, solves and
  Fock correction. Stream C: one-center terms. Joined by events before Pulay.
- Energy (helect) overlaps the Pulay commutator.
- D3H4 runs on its own stream for the whole SCF; its energy joins at the end.
- Direct mode: integral generation feeds the Fock contraction tile by tile (producer and
  consumer fused in one kernel, or two kernels linked through a ring buffer in device memory
  with stream events), so integrals never reach global memory.
- First iteration: integral generation (stored mode) overlaps with COSMO cavity setup.
- Multi-GPU: integrals and Fock work split by atom-pair ranges across devices; F reduced
  onto the device that runs the solve (or the multi-GPU solver distributes it).

MOZYME: Fock (pair kernels), far field (N-body) and COSMO run concurrently; diagg1 per
virtual LMO is fully parallel; diagg2 color classes run as a persistent kernel or a CUDA
graph.

CUDA graphs: once shapes are fixed after the first iteration, one iteration is captured as
a graph and replayed, removing launch overhead for the many small kernels.

## 6. VRAM residency policy

At start-up the run measures free memory on its device(s) and picks the highest tier that
fits, falling back step by step rather than failing:

1. Everything resident, integrals stored, deep Pulay history (96 GB cards).
2. Integrals stored, Pulay history trimmed (24 GB).
3. Direct SCF (integrals recomputed per iteration), history trimmed (16 GB).
4. Direct SCF and COSMO bmat recomputed or streamed in panels (8 GB).
5. CPU path.

The chosen tier is printed in the output, like the current `MAGMA:` line.

## 7. Validation

- Per kernel: against the CPU routine on randomized pair geometries for every pair class
  (H-H, sp-H, sp-sp, d-H, d-sp, d-d), including feathered and unfeathered, to ~1e-14
  relative.
- Whole SCF: the fork's regression set (water dimer, ethane, methanol, size ladder
  clusters, COSMO dimer and ionic clusters), the 749-atom globule, and whole PTP1B:
  heats of formation to the printed digits, AO populations to 1e-5 e, iteration counts
  reported.
- MOZYME: against CPU MOZYME at the study criterion (SCFCRT=1e-3), AO populations and
  energies to SCF tolerance (rotation order differs).
- Across machines: 2x RTX A4000 (a T400 in the same machine is never used),
  RTX 5060 (8 GB tier), RTX 5090 Laptop, RTX PRO 6000 Blackwell.

## 8. Staging (each stage measured on whole PTP1B before the next)

1. Conventional diagonalization without host overhead: device interface
   (`magma_dsyevd_gpu`), persistent workspaces, F/C resident. Expected: 25 s -> ~10 s per
   iteration on the RTX PRO 6000.
2. Density (cuBLAS dsyrk) and Pulay on the device, P and history resident.
3. Conventional Fock + one-center + COSMO addfck kernels, F built on the device. After this
   stage nothing but scalars crosses PCIe per iteration.
4. Integral kernels (HCORE) on the device, stored mode; then direct mode.
5. MOZYME: sparse Fock with device integrals, far field, COSMO (linear), density.
6. MOZYME: diagg1, then diagg2 with edge coloring, tidy/check on the device.
7. D3H4 energy and gradients; finite-difference gradients via the integral kernel.
8. CUDA graphs, multi-GPU work split, residency tiers.

## 9. Open questions

- Pulay depth vs iterations on the GPU: does a deeper history cut the 46 iterations?
- MOZYME on the GPU: is the CPU's damping/extrapolation state machine kept as is, or
  replaced once the per-iteration cost drops (more iterations become cheap)?
- MAGMA ILP64 build for systems beyond ~32,700 AOs on 96 GB cards.
- Build: CUDA sources (.cu) in a new `src/gpu/` directory compiled with nvcc as part of
  the existing CMake (CUDA language enabled only with the GPU option).
