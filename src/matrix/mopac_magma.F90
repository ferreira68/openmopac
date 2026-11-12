! Molecular Orbital PACkage (MOPAC)
! Copyright 2021 Virginia Polytechnic Institute and State University
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!    http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

! GPU dense linear algebra through MAGMA (built with -DMOPAC_MAGMA).
!
! MAGMA is called through bind(C) interfaces declared here, not through MAGMA's own Fortran
! modules, so a MAGMA library built with one gfortran version links into MOPAC built with another.
! The interfaces assume MAGMA's default LP64 build (magma_int_t = 32-bit int).
!
! A device is usable only if this MAGMA build has kernels for it: compute capability 8.6-8.9
! (sm_86 SASS) or 12.x (sm_120). Other devices (e.g. a T400, 7.5) are never used.
!
! Device selection (once per run, magma_setup):
!   NOGPU        - never use the GPU.
!   SETGPU=n     - one GPU: CUDA device n (1-based, as in the original MOPAC keyword).
!   otherwise    - MAGMA's multi-GPU routines use devices 0..ngpu-1, so ngpu is the number of
!                  usable devices at the start of the visible list (pick devices with
!                  CUDA_VISIBLE_DEVICES).
!                  With fewer than two such devices: one GPU, the usable device with the most
!                  free memory at start-up.
! Per call (magma_fits): a matrix of order n goes to the GPU only if n >= magma_min_n and it fits:
! device 0 (or the single device) needs 2*n**2 + 64*n doubles (magma_dsyevd's divide-and-conquer
! workspace is about 1.5*n**2 on the first device) and each further device 1.2*n**2/ngpu + 64*n,
! all with 10% headroom. Otherwise the CPU LAPACK path is used. magma_min_n is 1000.
!
! Without -DMOPAC_MAGMA this module compiles to stubs and the GPU is never used.

module mopac_magma
  use iso_c_binding, only : c_int, c_double, c_size_t
  implicit none
  private
  public :: magma_on, magma_device, magma_ngpu, magma_min_n, magma_calls, magma_cpu_fallbacks, magma_setup, magma_fits
  public :: magma_timing, magma_wall, magma_workspace, mg_work, mg_iwork
#ifdef MOPAC_MAGMA
  public :: MagmaUpper, MagmaLower, MagmaVec, magma_dsyevd, magma_dsyevd_m
#endif

  logical :: magma_on = .false.           ! GPU path available and enabled for this run
  integer :: magma_device = -1            ! 0-based CUDA device in use (device 0 when magma_ngpu > 1)
  integer :: magma_ngpu = 0               ! GPUs used per eigensolve: 1 = magma_dsyevd, > 1 = magma_dsyevd_m
  integer :: magma_min_n = 1000           ! smallest matrix order sent to the GPU
  integer :: magma_calls = 0              ! eigensolves done on the GPU
  integer :: magma_cpu_fallbacks = 0      ! GPU attempts that failed and were redone on the CPU
  logical :: magma_timing = .false.       ! TIMES keyword: print a per-call breakdown of the GPU eigensolve
  ! Persistent eigensolver workspaces: allocated once (grown when needed) instead of on every call,
  ! which for N = 12,362 is a 2.4 GB allocation and first touch per SCF iteration.
  double precision, allocatable :: mg_work(:)
  integer(c_int), allocatable :: mg_iwork(:)

#ifdef MOPAC_MAGMA
  integer(c_int), parameter :: MagmaUpper = 121, MagmaLower = 122, MagmaVec = 302   ! magma_types.h
  integer(c_int), parameter :: cudaDevAttrComputeCapabilityMajor = 75, cudaDevAttrComputeCapabilityMinor = 76

  interface
    integer(c_int) function magma_init() bind(C, name="magma_init")
      import :: c_int
    end function magma_init
    subroutine magma_setdevice(dev) bind(C, name="magma_setdevice")
      import :: c_int
      integer(c_int), value :: dev
    end subroutine magma_setdevice
    integer(c_int) function magma_getdevice_arch() bind(C, name="magma_getdevice_arch")
      import :: c_int
    end function magma_getdevice_arch
    integer(c_int) function magma_dsyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info) &
        bind(C, name="magma_dsyevd")
      import :: c_int, c_double
      integer(c_int), value :: jobz, uplo, n, lda, lwork, liwork
      real(c_double) :: a(*), w(*), work(*)
      integer(c_int) :: iwork(*), info
    end function magma_dsyevd
    integer(c_int) function magma_dsyevd_m(ngpu, jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info) &
        bind(C, name="magma_dsyevd_m")
      import :: c_int, c_double
      integer(c_int), value :: ngpu, jobz, uplo, n, lda, lwork, liwork
      real(c_double) :: a(*), w(*), work(*)
      integer(c_int) :: iwork(*), info
    end function magma_dsyevd_m
    integer(c_int) function cudaDeviceGetAttribute(val, attr, dev) bind(C, name="cudaDeviceGetAttribute")
      import :: c_int
      integer(c_int) :: val
      integer(c_int), value :: attr, dev
    end function cudaDeviceGetAttribute
    integer(c_int) function cudaGetDeviceCount(count) bind(C, name="cudaGetDeviceCount")
      import :: c_int
      integer(c_int) :: count
    end function cudaGetDeviceCount
    integer(c_int) function cudaSetDevice(dev) bind(C, name="cudaSetDevice")
      import :: c_int
      integer(c_int), value :: dev
    end function cudaSetDevice
    integer(c_int) function cudaMemGetInfo(free_bytes, total_bytes) bind(C, name="cudaMemGetInfo")
      import :: c_int, c_size_t
      integer(c_size_t) :: free_bytes, total_bytes
    end function cudaMemGetInfo
  end interface
#endif

contains

  subroutine magma_setup(keywrd, iw)
    ! Choose and initialize the GPU once per run; report the choice in the output file.
    character(len=*), intent(in) :: keywrd
    integer, intent(in) :: iw
#ifdef MOPAC_MAGMA
    integer(c_int) :: ndev, dev, stat, major, minor
    integer(c_size_t) :: free_b, total_b, best_free
    integer :: i, req, maxgpu, lead
    logical :: usable(0:63), conv
    logical, save :: done = .false.
    double precision, external :: reada

    if (done) return
    done = .true.
    conv = index(keywrd, " MOZYME") == 0   ! a MOZYME job has its own execution-path line (acc_path_start)
    magma_on = .false.
    magma_timing = index(keywrd, " TIMES") /= 0
    maxgpu = 64
    if (index(keywrd, " NOGPU") /= 0) then
      if (conv) write (iw, '(a)') " Using CPU execution"
      write (iw, '(a)') " dense eigensolves on CPU (LAPACK; keyword NOGPU)"
      return
    end if
    ndev = 0
    stat = cudaGetDeviceCount(ndev)
    if (stat /= 0 .or. ndev < 1) then
      if (conv) write (iw, '(a)') " Using CPU execution"
      write (iw, '(a,i0,a)') " dense eigensolves on CPU (LAPACK; no usable CUDA device, cudaGetDeviceCount status ", stat, &
        ")"
      return
    end if
    usable = .false.
    do i = 0, min(ndev, 64) - 1
      if (cudaDeviceGetAttribute(major, cudaDevAttrComputeCapabilityMajor, int(i, c_int)) /= 0) cycle
      if (cudaDeviceGetAttribute(minor, cudaDevAttrComputeCapabilityMinor, int(i, c_int)) /= 0) cycle
      usable(i) = (major == 8 .and. minor >= 6) .or. major == 12
    end do
    lead = 0
    do i = 0, min(ndev, 64) - 1
      if (.not. usable(i)) exit
      lead = lead + 1
    end do
    req = -1
    i = index(keywrd, " SETGPU=")
    if (i /= 0) then
      req = nint(reada(keywrd, i)) - 1
      if (req < 0 .or. req >= ndev) then
        write (iw, '(a,i0,a,i0,a)') " dense eigensolves on GPU chosen automatically (SETGPU=", req + 1, &
          " is not one of the ", ndev, &
          " visible devices)"
        req = -1
      end if
    end if
    if (req >= 0) then
      if (.not. usable(req)) then
        if (conv) write (iw, '(a)') " Using CPU execution"
        write (iw, '(a,i0,a)') " dense eigensolves on CPU (LAPACK; SETGPU=", req + 1, &
          " is not a device this MAGMA build supports)"
        return
      end if
      dev = req
      magma_ngpu = 1
    else if (min(lead, maxgpu) >= 2) then
      dev = 0
      magma_ngpu = min(lead, maxgpu)
    else
      magma_ngpu = 1
      dev = -1
      best_free = 0
      do i = 0, ndev - 1
        if (.not. usable(i)) cycle
        if (cudaSetDevice(int(i, c_int)) /= 0) cycle
        if (cudaMemGetInfo(free_b, total_b) /= 0) cycle
        if (free_b > best_free) then
          best_free = free_b
          dev = i
        end if
      end do
      if (dev < 0) then
        if (conv) write (iw, '(a)') " Using CPU execution"
        write (iw, '(2a)') " dense eigensolves on CPU (LAPACK; no visible device is supported by this MAGMA build,", &
          " which needs compute capability 8.6-8.9 or 12.x)"
        return
      end if
    end if
    if (cudaSetDevice(dev) /= 0) then
      if (conv) write (iw, '(a)') " Using CPU execution"
      write (iw, '(a,i0,a)') " dense eigensolves on CPU (LAPACK; cudaSetDevice(", dev, ") failed)"
      return
    end if
    if (magma_init() /= 0) then
      if (conv) write (iw, '(a)') " Using CPU execution"
      write (iw, '(a)') " dense eigensolves on CPU (LAPACK; magma_init failed)"
      return
    end if
    call magma_setdevice(dev)
    stat = cudaMemGetInfo(free_b, total_b)
    magma_device = dev
    magma_on = .true.
    if (magma_ngpu > 1) then
      if (conv) write (iw, '(a)') " Using GPU/CPU hybrid execution"
      write (iw, '(a,i0,a,i0,a,i0,a)') " dense eigensolves of order >= ", magma_min_n, &
        " on GPU when they fit (MAGMA multi-GPU magma_dsyevd_m, ", &
        magma_ngpu, " GPUs, CUDA devices 0-", magma_ngpu - 1, ")"
    else
      if (conv) write (iw, '(a)') " Using GPU/CPU hybrid execution"
      write (iw, '(a,i0,a,i0,a,i0,a,i0,a,f7.1,a,f7.1,a)') " dense eigensolves of order >= ", magma_min_n, &
        " on GPU when they fit (MAGMA, CUDA device ", dev, " of ", ndev, &
        ", arch ", magma_getdevice_arch(), ", ", dble(free_b)/1024.d0**3, " of ", dble(total_b)/1024.d0**3, " GiB free)"
    end if
#else
    if (.false.) write (iw, '(a)') keywrd   ! no GPU support compiled in
#endif
  end subroutine magma_setup

  logical function magma_fits(n)
    ! True if magma_dsyevd of order n should run on the GPU now.
    integer, intent(in) :: n
#ifdef MOPAC_MAGMA
    integer(c_size_t) :: free_b, total_b
    double precision :: need
    integer :: i
    integer(c_int) :: stat
    magma_fits = .false.
    if (.not. magma_on .or. n < magma_min_n) return
    if (magma_ngpu <= 1) then
      if (cudaMemGetInfo(free_b, total_b) /= 0) return
      need = 8.d0*(2.d0*dble(n)*dble(n) + 64.d0*dble(n))*1.1d0
      magma_fits = need < dble(free_b)
      return
    end if
    do i = magma_ngpu - 1, 0, -1        ! ends on device 0, MAGMA's base device
      if (cudaSetDevice(int(i, c_int)) /= 0) return
      if (cudaMemGetInfo(free_b, total_b) /= 0) return
      if (i == 0) then
        need = 8.d0*(2.d0*dble(n)*dble(n) + 64.d0*dble(n))*1.1d0
      else
        need = 8.d0*(1.2d0*dble(n)*dble(n)/dble(magma_ngpu) + 64.d0*dble(n))*1.1d0
      end if
      if (need >= dble(free_b)) then
        if (i /= 0) stat = cudaSetDevice(0_c_int)   ! leave MAGMA's base device current
        return
      end if
    end do
    magma_fits = .true.
#else
    magma_fits = .false.
    if (n < 0) magma_fits = .false.
#endif
  end function magma_fits

  double precision function magma_wall()
    ! Wall-clock seconds (for the TIMES breakdown).
    integer(8) :: count, rate
    call system_clock(count, rate)
    magma_wall = dble(count)/dble(rate)
  end function magma_wall

  subroutine magma_workspace(lwork, liwork, ok)
    ! Make sure the persistent workspaces hold at least lwork doubles and liwork integers.
    integer, intent(in) :: lwork, liwork
    logical, intent(out) :: ok
    integer :: stat
    ok = .true.
    if (allocated(mg_work)) then
      if (size(mg_work) < lwork) deallocate (mg_work)
    end if
    if (.not. allocated(mg_work)) then
      allocate (mg_work(lwork), stat=stat)
      if (stat /= 0) then
        ok = .false.
        return
      end if
    end if
    if (allocated(mg_iwork)) then
      if (size(mg_iwork) < liwork) deallocate (mg_iwork)
    end if
    if (.not. allocated(mg_iwork)) then
      allocate (mg_iwork(liwork), stat=stat)
      if (stat /= 0) ok = .false.
    end if
  end subroutine magma_workspace

end module mopac_magma
