# Molecular Orbital PACkage (MOPAC)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.08025/status.svg)](https://doi.org/10.21105/joss.08025)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6511958.svg)](https://doi.org/10.5281/zenodo.6511958)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/mopac/badges/version.svg)](https://anaconda.org/conda-forge/mopac)
![build](https://github.com/openmopac/mopac/actions/workflows/CI.yaml/badge.svg)
[![codecov](https://codecov.io/gh/openmopac/mopac/branch/main/graph/badge.svg?token=qM2KeRvw06)](https://codecov.io/gh/openmopac/mopac)

This is the official repository of the modern open-source version of MOPAC, 
which is a direct continuation of the commercial development and distribution of MOPAC that ended at MOPAC 2016.

[![mopac_at_molssi](.github/mopac_at_molssi.png)](https://molssi.org)

MOPAC is actively maintained and curated by the [Molecular Sciences Software Institute (MolSSI)](https://molssi.org).

For detailed information about MOPAC, see its [website](https://openmopac.github.io). A brief summary for developers is provided below.

## About

MOPAC is a popular semiempirical quantum chemistry package that was first released in 1983.
MOPAC can perform quantum mechanical calculations of molecules and materials, much like *ab initio* quantum chemistry packages such as [Gaussian](https://www.gaussian.com).
Semiempirical calculations are around 1000x faster than *ab initio* calculations, but they are usually less accurate and less predictive because they rely on
simplified models fit to experimental data rather than more complicated and comprehensive theories.
The speed and ease-of-use of MOPAC make it a good teaching tool, either to teach physical chemistry concepts to students or to prepare for the use
of more expensive quantum chemistry software. It can also be useful for high-throughput calculations in screening and informatics applications.

## Building

Pre-built versions of MOPAC on Linux, Mac, and Windows are available with every [GitHub Release](https://github.com/openmopac/mopac/releases)
and also available on the Conda package manager using the command:
```
conda install -c conda-forge mopac
```

MOPAC can be built using its CMake build system. This repository has a [GitHub Actions workflow](https://github.com/openmopac/mopac/blob/main/.github/workflows/CI.yaml)
to build MOPAC, and you can build it locally using the standard CMake out-of-source invocation:
```
mkdir build
cd build
cmake ..
make
```
MOPAC will build without any other CMake options if a Fortran compiler, BLAS/LAPACK, Python 3, and NumPy are found in the software environment.
To build with optional [MolSSI Driver Interface(MDI)](https://molssi-mdi.github.io/MDI_Library/) engine support, use the CMake command-line option `-DMDI=ON`.

## GPU acceleration (openmopac-gpu)

This fork adds NVIDIA GPU acceleration for MOZYME, the linear-scaling SCF used for proteins and other large systems.
On a whole protein (PTP1B, about 4,900 atoms, PM6, `CUTOFF=100`) the MOZYME SCF runs about 13x faster on an RTX 5090
Laptop GPU than the CPU-only program, with the same result to the precision of the output.

### Building

The GPU code is written in OpenACC and needs the NVIDIA HPC SDK compiler (`nvfortran`). Full instructions,
tested versions, run-time libraries and testing: [BUILDING_GPU.md](BUILDING_GPU.md). In short:
```
FC=nvfortran CC=nvc cmake .. -DGPU_OPENACC=ON -DGPU_OPENACC_ARCH="86;120" \
    -DCMAKE_BUILD_TYPE=None -DCMAKE_Fortran_FLAGS="-O2" -DMOPAC_LINK="/path/to/libopenblas.so.0"
make
```
`GPU_OPENACC_ARCH` lists the compute capabilities to build device code for (`86` covers sm_86 to sm_89, e.g. RTX 30/40-series
and A40/L40S; `120` covers RTX 50-series and RTX PRO Blackwell; add `80` for A100 or `90` for H100). The program then needs
the NVIDIA driver at run time. Builds without `GPU_OPENACC` are unchanged CPU builds.

Optionally, `-DMAGMA_ROOT=<static MAGMA install> -DCUDA_STATIC_LIB=<CUDA lib64>` adds the MAGMA GPU eigensolver for the
dense diagonalizations of conventional (non-MOZYME) SCF calculations.

### What runs on the GPU

In a GPU build, MOZYME uses the GPU automatically whenever a usable device is present: the Fock matrix build, the
density matrix, the occupied-virtual screening and rotation setup of DIAGG1, and the electronic energy. The remaining
steps (one-electron matrix, DIAGG2, convergence control) run on the CPU with OpenMP threads. If the device has too little
memory for all of this, the parts that do not fit stay on the CPU. Conventional SCF calculations run on the CPU.

Every output file states the execution path, in lines that start with `MOPAC execution path:`: **full GPU**,
**GPU/CPU hybrid** (with what runs on the CPU and why), or **CPU** (and why: no GPU visible, `NOGPU`, a device the
build has no code for, not enough device memory, ...). Example:
```
 MOPAC execution path: MOZYME on the GPU (device 1, NVIDIA GeForce RTX 5090 Laptop GPU, compute capability 12.0); ...
 MOPAC execution path: full GPU (every MOZYME step with GPU code: Fock build, density, DIAGG1, energy)
```
The keywords `TIMES` or `DEBUG` add the details: device-memory plan, kernel reservations, transfer settings.

### Keywords and environment variables

| Name | Kind | Effect |
|---|---|---|
| `NOGPU` | keyword | Run everything on the CPU. |
| `SETGPU=n` | keyword | Use visible GPU number `n` (1, 2, ...). By default, the first visible GPU the build has device code for is used. |
| `NOFEATHER` | keyword | Keep the full NDDO two-electron integrals under MOZYME. By default MOZYME blends the integrals of PM6 and later methods into point charges at long range ("feathering"), which changes the energy expression relative to the conventional SCF. |
| `NOGASSCF` | keyword | With COSMO, start the SCF in solvent instead of converging a gas-phase SCF first. |
| `MOPAC_ACC_FOCK`, `MOPAC_ACC_DIAGG`, `MOPAC_ACC_DENS` | environment | `0` keeps that step on the CPU; `test` runs it on both and writes the largest difference to the output on every call (validation). |
| `MOPAC_ACC_MEM_LIMIT` | environment | Cap on the device memory MOZYME may use, in bytes or with a `KB`, `MB` or `GB` suffix (for a GPU shared with other jobs). |

GPU and CPU results agree to the numerical precision of the algorithms (for a whole protein, heats of formation to
within about 1e-4 kcal/mol).

## Usage

While MOPAC is often used through its integration with other software such as [WebMO](https://www.webmo.net), it is primarily a command-line program
that operates through input and output files. Recent versions of MOPAC also has an application programming interface (API) for a subset of functionality.

Examples of command-line and API usage are provided in the [`/examples` folder](https://github.com/openmopac/mopac/tree/main/examples) of this repository.

## Citation

The following paper should be cited by scientific publications that use an open-source version of the MOPAC software package:

J. E. Moussa and J. J. P. Stewart, MOPAC: An open-source semiempirical molecular orbital program, *J. Open Source Softw.* **2026**, *11*(119), 8025.
DOI:[10.21105/joss.08025](https://doi.org/10.21105/joss.08025)

Alternatively, the open-source MOPAC software package can be cited directly through a software citation to its archival copy on Zenodo at
DOI:[10.5281/zenodo.6511958](https://doi.org/10.5281/zenodo.6511958).
