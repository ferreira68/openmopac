# Building openmopac-gpu

This guide covers the GPU build of this fork: MOZYME with the Fock build, density matrix, DIAGG1 and the
electronic energy on an NVIDIA GPU (OpenACC). A CPU-only build is the standard MOPAC build (see `README.md`)
and is unaffected by any of the options below.

## Requirements

| Component | Tested version | Notes |
|---|---|---|
| NVIDIA HPC SDK (`nvfortran`, `nvc`) | 26.9 (bundled CUDA 13.3) | Required for `GPU_OPENACC=ON`. Free download from NVIDIA. |
| NVIDIA driver | 595.84 – 595.99 | CUDA 13 needs an R580 or newer driver. |
| CMake | 3.28 | 3.20 or newer (CMake must identify the compiler as `NVHPC`). |
| BLAS/LAPACK | OpenBLAS 0.3.34 (OpenMP) | Any BLAS/LAPACK works; give it with `MOPAC_LINK` (below). |
| Python 3 + NumPy | 3.12 / 2.4 | Only for the test suite (`-DTESTS=ON`). |
| gfortran (CPU-only builds) | 13.3 | Unchanged from upstream. |

Tested GPUs: RTX 5090 Laptop and RTX 5060 (compute capability 12.0), RTX PRO 6000 Blackwell Server
Edition (12.0), RTX A4000 (8.6).

## Build

```bash
git clone https://github.com/ferreira68/openmopac.git
cd openmopac
git checkout openmopac-gpu-v1.0
mkdir build && cd build
FC=nvfortran CC=nvc cmake .. \
    -DGPU_OPENACC=ON \
    -DGPU_OPENACC_ARCH="86;120" \
    -DCMAKE_BUILD_TYPE=None -DCMAKE_Fortran_FLAGS="-O2" \
    -DMOPAC_LINK="/path/to/libopenblas.so.0 -Wl,-rpath,/path/to"
make -j 16
```

Notes on the options:

- **`GPU_OPENACC=ON`** compiles the OpenACC code (`-acc=gpu`) and also adds `-Mbackslash`, which nvfortran
  needs for MOPAC's string handling. OpenMP (`THREADS` keyword) stays on by default (`THREADS_KEYWORD=ON`).
- **`GPU_OPENACC_ARCH`** lists the compute capabilities to generate device code for, as two- or three-digit
  numbers (`86` = sm_86, `120` = sm_120). A device can run code built for the same major version and an equal
  or lower minor version, so:

  | GPU family | Compute capability | Include |
  |---|---|---|
  | A100 | 8.0 | `80` |
  | RTX 30-series, A40, RTX A4000–A6000 | 8.6 | `86` |
  | RTX 40-series, L4, L40S, RTX 6000 Ada | 8.9 | `86` (or `89`) |
  | H100, H200 | 9.0 | `90` |
  | B200 | 10.0 | `100` |
  | RTX 50-series, RTX PRO 6000 Blackwell | 12.0 | `120` |

  At run time, a device without matching code is not used: MOZYME runs on the CPU and the output says why.
  Only `86` and `120` have been tested.
- **Optimization level.** The validated configuration is `CMAKE_BUILD_TYPE=None` with `-O2`.
  `CMAKE_BUILD_TYPE=Release` gives nvfortran `-fast -O3`, which has not been validated for this code.
- **`MOPAC_LINK`.** Give the BLAS/LAPACK library as a full path. nvfortran's linker driver rewrites the
  GNU form `-l:libopenblas.so.0` into `-llibopenblas.so.0`, which fails.
- **Rebuilding.** If a rebuild stops with `ld: input file 'libmopac.so.2' is the same as output file`,
  delete `libmopac.so.2` from the build directory and run `make` again.

### Optional: MAGMA eigensolver for conventional SCF

Conventional (non-MOZYME) SCF calculations can use the MAGMA GPU eigensolver for their dense diagonalizations.
This needs a static, position-independent MAGMA build and the static CUDA libraries:

```bash
cmake .. ... -DMAGMA_ROOT=/path/to/magma-static -DCUDA_STATIC_LIB=/path/to/cuda/lib64
```

The tested MAGMA was built from the MAGMA repository (commit 64a52642, after v2.10.0) as a static, PIC library
for sm_86 and sm_120 with a full CUDA 13.2 toolkit. MAGMA uses 32-bit integers, which limits it to about 32,700
basis functions. MAGMA is not part of the v1.0 test matrix.

## Run-time requirements

- **NVIDIA driver** (R580 or newer). The program loads the driver's `libcuda` at run time and links no CUDA
  toolkit library; it has, however, only been tested on machines that also had a CUDA toolkit installed.
- **HPC SDK runtime libraries.** The program links `libacchost`, `libaccdevice`, `libaccdevaux`, `libnvf`,
  `libnvomp`, `libnvc`, `libnvcpumath` and `libnvhpcatm`. On machines without the HPC SDK, ship them with the
  program; the SDK provides the redistributable set in `Linux_x86_64/<version>/REDIST/compilers/lib`. Point
  `LD_LIBRARY_PATH` (or an rpath) at them, together with the BLAS/LAPACK library.
- **`FORTRANOPT=crlf`** in the environment if input files may have Windows (CRLF) line endings. Without it, the
  nvfortran runtime does not read such files (the MOPAC test `erratic-3` uses one).

## Controlling and checking the GPU use

- MOZYME uses the GPU automatically: by default the first visible GPU the build has device code for. `NOGPU`
  keeps everything on the CPU; `SETGPU=n` picks visible GPU `n` (1, 2, ...); `CUDA_VISIBLE_DEVICES` restricts
  the visible GPUs.
- `MOPAC_ACC_MEM_LIMIT=8GB` (or `KB`, `MB`, bytes) caps the device memory MOZYME may use on a shared GPU.
- `MOPAC_ACC_FOCK`, `MOPAC_ACC_DIAGG`, `MOPAC_ACC_DENS`: `0` keeps that step on the CPU; `test` runs it on
  both and writes the largest difference on every call (for validation).
- Every output file states the execution path in lines starting with `MOPAC execution path:` (full GPU,
  GPU/CPU hybrid with what runs on the CPU and why, or CPU and why). `TIMES` or `DEBUG` add the device-memory
  plan and other details, in lines starting with `MOPAC GPU`.

## Testing

```bash
cmake .. -DTESTS=ON -DPython3_EXECUTABLE=$(which python3)
make -j 16
FORTRANOPT=crlf ctest -j 8                                                          # GPU on (default)
FORTRANOPT=crlf MOPAC_ACC_FOCK=0 MOPAC_ACC_DIAGG=0 MOPAC_ACC_DENS=0 ctest -j 8     # GPU steps off
```

Expected: all 125 tests pass in both runs (and in a gfortran CPU-only build).

The test comparison skips lines that contain `MOPAC`, so the execution-path and GPU lines do not affect it.
GPU and CPU heats of formation agree within the test tolerance (0.01 kcal/mol): for example, 6e-4 kcal/mol for
the `ALLBONDS` test molecule and 3e-5 kcal/mol for a whole protein (PTP1B, PM6, MOZYME).
