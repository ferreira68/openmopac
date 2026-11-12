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

! OpenACC (GPU) version of the two-center part of the MOZYME Fock matrix (fz2n), direct mode.
!
! Integrals are generated and contracted with the density on the device, pair by pair, and never
! stored. Covered: every atom pair treated by point charges (nijbo < 0, monopole and dipole terms)
! and every NDDO pair (nijbo >= 0) in which neither atom has d orbitals. NDDO pairs that involve
! an atom with d orbitals, the one-center terms (fock1_for_MOZYME) and COSMO (addfckz) stay on the
! CPU in fz2n.
!
! One gang per atom A accumulates every contribution to A's diagonal block; the gang of the
! higher-numbered atom of an NDDO pair also writes that pair's exchange (off-diagonal) block.
! Every Fock element therefore has a single writer: no atomic updates, and the result is
! deterministic. Each NDDO pair's integrals are evaluated twice, once by each atom's gang.
! The arithmetic of each term follows fz2n, jab_for_MOZYME and kab_for_MOZYME.
!
! Enabled by the environment variable MOPAC_ACC_FOCK: "test" also runs the CPU code and writes
! the largest difference to the output for every Fock build; any other value uses the GPU.
! Without OpenACC, acc_fock_mode returns 0 and the GPU routines are empty.

module acc_mozyme_C
! Device-side MOZYME data shared by the OpenACC routines (acc_fz2n, acc_diagg1_core):
! - nij: a copy of nijbo, present on the device, refreshed when the geometry or the set of
!   interactions changes;
! - fd, pd: device buffers (acc_malloc) of length mpack for the Fock and density matrices.
!   They are filled by explicit copies from the host arrays, so a host array may be deallocated
!   at any time without leaving a stale device mapping. fd_ok / pd_ok say that fd / pd hold the
!   current common_arrays f / p: set by acc_fz2n after a complete Fock build, cleared at the start
!   of every fz2n call and after diagg1 has used them (the density changes after diagg1).
! - pacc: the fixed-point accumulator of the GPU density build (length mpack, 64-bit integers).
!
! Device memory plan (acc_mozyme_plan, at every rebuild of nij): the buffers above are released,
! the free device memory is queried, the memory each GPU path needs is computed from mpack, numat,
! the number of NDDO pairs and the LMO array sizes, and the largest set of the requested paths
! that fits (with headroom) is enabled, shedding first the NDDO pairs with d orbitals (kernel stacks,
! below), then the density (its accumulator), then diagg1, then the Fock build; whatever is not
! enabled runs on the CPU. The decision is written to the output. MOPAC_ACC_MEM_LIMIT (bytes, or
! with suffix KB/MB/GB) caps the memory the plan may use, and MOPAC_ACC_MEM_LEVEL (1-6) sets the
! first level tried, to test the fallbacks on a large card. A failed acc_malloc later disables the
! GPU paths (CPU from then on) instead of stopping the program.
!
! Local memory: CUDA reserves a kernel's per-thread stack for every thread slot of the device
! (SMs x maximum resident threads per SM), whatever the launch size, and keeps the reservation.
! Per-thread stacks (Nsight Compute, "Stack Size", nvfortran 26.9, this code): rotate_sp (s,p
! pairs) 6,720 bytes; rotate_spd (pairs with one s,p,d atom) 20,912; the general rotate
! (pairs of two s,p,d atoms, "d,d") 71,168. On an RTX 5090 Laptop (82 SMs x 1,536) the d,d kernel
! alone reserves 8.4 GB. The plan includes these terms; the first launch of each pair kernel
! measures the actual reservation and reports it if it differs from the prediction by more than
! 10 % (e.g. after a code change that alters the stack).
! Levels (shedding order, by benefit per byte): 1 everything on the GPU; 2 d,d pairs on the CPU;
! 3 all pairs with d orbitals on the CPU; 4 also the density; 5 also diagg1 (Fock only); 6 nothing.
! Level 1 (d,d pairs on the GPU) is decided at run time, because its value depends on the system,
! the CPU and the GPU: the d,d kernel reserves gigabytes (stack x thread slots), while on the CPU
! (acc_fz2n_cpu_dpairs, OpenMP) the d,d pairs may cost little. The plan starts at level 2 unless the
! reservation is already held (an earlier geometry); then, at level 2, the first Fock builds measure
! the CPU d,d pass (t_dd_cpu), the time per SCF iteration (between Fock builds, t_iter) and the GPU
! s,p-d kernel time per pair, from which the d,d kernel time is estimated (x k_dd: d,d pairs cost
! about 11 times an s,p-d pair on the GPU, Nsight Systems, RTX 5090 Laptop, 19,900 d,d / 360,000
! s,p-d pairs). The d,d pairs move to the GPU (acc_dd_decide) if the estimated saving exceeds f_dd
! (5 %) of an iteration and the reservation fits in the free device memory with the plan's headroom.
! One decision per geometry. MOPAC_ACC_MEM_LEVEL=1 forces level 1.
  use iso_c_binding, only : c_ptr, c_null_ptr, c_size_t, c_long_long, c_int, c_char, c_funptr
  implicit none
  integer, allocatable :: nij(:,:)
  integer :: key_numcal = -1, key_mpack = -1, key_numat = -1
  integer :: nij_gen = 0   ! incremented on every rebuild; consumers compare with the generation they used
  type(c_ptr) :: fd_c = c_null_ptr, pd_c = c_null_ptr, pacc_c = c_null_ptr
  integer :: dev_len = 0, pacc_len = 0
  logical :: fd_ok = .false., pd_ok = .false.
  logical :: allow_fock = .false., allow_diagg = .false., allow_dens = .false.
  logical :: allow_sdgpu = .false., allow_ddgpu = .false.   ! s,p-d and d,d NDDO pairs on the GPU
  integer(8), parameter :: stack_rotate = 71168_8, stack_rotate_sp = 6720_8, stack_rotate_spd = 20912_8
  double precision, parameter :: k_dd = 11.d0, f_dd = 0.05d0
  logical :: dd_pending = .false.            ! level 2 by policy: decide after measuring (acc_dd_decide)
  double precision :: t_dd_cpu = -1.d0, t_sd_gpu = -1.d0, t_iter = -1.d0   ! seconds, last Fock build
  integer(8) :: t_last_build = -1, plan_headroom = 0
  integer(8) :: lmem_sp = 0, lmem_sd = 0, lmem_dd = 0, lmem_held = 0   ! predicted reservations; largest so far
  integer :: n_sdpairs = 0, n_ddpairs = 0                 ! NDDO pairs of each kind with d orbitals (acc_fz2n)
! The NDDO pairs with d orbitals (built by acc_fz2n, used by its kernels and by acc_fz2n_cpu_dpairs):
! dp_i > dp_j, their kind dp_cls (acc_pair_class: 1 s,p-d, 2 d,d), the indices of each kind (dp_isd,
! dp_idd), per atom the list of those pairs (dp_list from dp_start: +k, the atom is dp_i(k); -k, it is
! dp_j(k)), in increasing k, and per pair the contributions to the two diagonal blocks (dp_ci for
! dp_i, dp_cj for dp_j).
  integer :: dp_nd = 0, dp_nsd = 0, dp_ndd = 0
  integer, allocatable :: dp_i(:), dp_j(:), dp_cls(:), dp_isd(:), dp_idd(:), dp_start(:), dp_list(:)
  double precision, allocatable :: dp_ci(:,:), dp_cj(:,:)
  integer(8) :: held_lists = 0, held_cand = 0   ! device bytes held by acc_fz2n's pair lists, diagg1's buffers
  interface
    type(c_ptr) function c_acc_malloc(bytes) bind(C, name="acc_malloc")
      import :: c_ptr, c_long_long
      integer(c_long_long), value :: bytes
    end function c_acc_malloc
    subroutine c_acc_free(dev) bind(C, name="acc_free")
      import :: c_ptr
      type(c_ptr), value :: dev
    end subroutine c_acc_free
    subroutine c_acc_memcpy_to_device(dev, host, bytes) bind(C, name="acc_memcpy_to_device")
      import :: c_ptr, c_long_long
      type(c_ptr), value :: dev, host
      integer(c_long_long), value :: bytes
    end subroutine c_acc_memcpy_to_device
    subroutine c_acc_memcpy_from_device(host, dev, bytes) bind(C, name="acc_memcpy_from_device")
      import :: c_ptr, c_long_long
      type(c_ptr), value :: host, dev
      integer(c_long_long), value :: bytes
    end subroutine c_acc_memcpy_from_device
    integer(c_size_t) function c_acc_get_property(devnum, devtype, property) bind(C, name="acc_get_property")
! OpenACC 2.6 API (openacc.h): acc_device_nvidia = 4; acc_property_memory = 1, acc_property_free_memory = 2.
      import :: c_size_t, c_int
      integer(c_int), value :: devnum, devtype, property
    end function c_acc_get_property
    integer(c_int) function c_acc_get_device_num(devtype) bind(C, name="acc_get_device_num")
      import :: c_int
      integer(c_int), value :: devtype
    end function c_acc_get_device_num
    type(c_ptr) function c_dlopen(name, flag) bind(C, name="dlopen")
      import :: c_ptr, c_char, c_int
      character(kind=c_char), dimension(*) :: name
      integer(c_int), value :: flag
    end function c_dlopen
    type(c_funptr) function c_dlsym(handle, name) bind(C, name="dlsym")
      import :: c_ptr, c_funptr, c_char
      type(c_ptr), value :: handle
      character(kind=c_char), dimension(*) :: name
    end function c_dlsym
  end interface
  abstract interface
    integer(c_int) function cu_device_get(device, ordinal) bind(C)     ! CUresult cuDeviceGet(CUdevice*, int)
      import :: c_int
      integer(c_int) :: device
      integer(c_int), value :: ordinal
    end function cu_device_get
    integer(c_int) function cu_device_get_attribute(pi, attrib, device) bind(C)  ! cuDeviceGetAttribute(int*, attr, CUdevice)
      import :: c_int
      integer(c_int) :: pi
      integer(c_int), value :: attrib, device
    end function cu_device_get_attribute
  end interface
contains
  logical function acc_sp_atom(z, io)
! An atom (atomic number z, io orbitals) with an s or s,p basis and no d-orbital terms.
    use parameters_C, only : dorbs, iod
    integer, intent(in) :: z, io
    acc_sp_atom = (io == 1 .or. io == 4) .and. .not. dorbs(z) .and. iod(z) == 0
  end function acc_sp_atom

  integer function acc_pair_class(za, ia, zb, ib)
! Kind of an atom pair for the GPU Fock build, from the atomic numbers and orbital counts:
! 0: both atoms s or s,p (rotate_sp); 1: one s,p,d atom with an s or s,p atom (rotate_spd);
! 2: any other pair (general rotate on the GPU only in the d,d kernel).
    integer, intent(in) :: za, ia, zb, ib
    logical :: spa, spb
    spa = acc_sp_atom(za, ia)
    spb = acc_sp_atom(zb, ib)
    if (spa .and. spb) then
      acc_pair_class = 0
    else if ((ia == 9 .and. spb) .or. (ib == 9 .and. spa)) then
      acc_pair_class = 1
    else
      acc_pair_class = 2
    end if
  end function acc_pair_class

  logical function acc_pair_on_gpu(za, ia, zb, ib)
! Whether the GPU Fock build (as planned) computes this NDDO pair; if not, fz2n does it on the CPU.
    integer, intent(in) :: za, ia, zb, ib
    select case (acc_pair_class(za, ia, zb, ib))
    case (0)
      acc_pair_on_gpu = .true.
    case (1)
      acc_pair_on_gpu = allow_sdgpu
    case default
      acc_pair_on_gpu = allow_ddgpu
    end select
  end function acc_pair_on_gpu

  logical function acc_mozyme_buffers(n)
! Makes fd and pd hold at least n doubles. On failure the GPU paths are disabled (.false.).
    use iso_c_binding, only : c_associated
    integer, intent(in) :: n
    acc_mozyme_buffers = .false.
#ifdef _OPENACC
    acc_mozyme_buffers = .true.
    if (n <= dev_len) return
    if (c_associated(fd_c)) call c_acc_free(fd_c)
    if (c_associated(pd_c)) call c_acc_free(pd_c)
    fd_c = c_null_ptr
    pd_c = c_null_ptr
    dev_len = 0
    fd_ok = .false.
    pd_ok = .false.
    fd_c = c_acc_malloc(int(n, c_long_long)*8_c_long_long)
    pd_c = c_acc_malloc(int(n, c_long_long)*8_c_long_long)
    if (.not. c_associated(fd_c) .or. .not. c_associated(pd_c)) then
      call acc_mozyme_give_up('allocating the device F and P buffers')
      acc_mozyme_buffers = .false.
      return
    end if
    dev_len = n
#endif
  end function acc_mozyme_buffers

  logical function acc_mozyme_pacc(n)
! Makes the density accumulator hold at least n 64-bit integers. On failure the GPU density is disabled.
    use iso_c_binding, only : c_associated
    use chanel_C, only : iw
    integer, intent(in) :: n
    acc_mozyme_pacc = .false.
#ifdef _OPENACC
    acc_mozyme_pacc = .true.
    if (n <= pacc_len) return
    if (c_associated(pacc_c)) call c_acc_free(pacc_c)
    pacc_len = 0
    pacc_c = c_acc_malloc(int(n, c_long_long)*8_c_long_long)
    if (.not. c_associated(pacc_c)) then
      pacc_c = c_null_ptr
      allow_dens = .false.
      write (iw, '(a)') ' ACC MOZYME: device memory exhausted allocating the density accumulator; density on the CPU'
      acc_mozyme_pacc = .false.
      return
    end if
    pacc_len = n
#endif
  end function acc_mozyme_pacc

  subroutine acc_mozyme_give_up(what)
! A device allocation failed: release what this module holds and run every MOZYME step on the CPU.
    use iso_c_binding, only : c_associated
    use chanel_C, only : iw
    character(len=*), intent(in) :: what
#ifdef _OPENACC
    call acc_mozyme_release()
    allow_fock = .false.
    allow_diagg = .false.
    allow_dens = .false.
    write (iw, '(3a)') ' ACC MOZYME: device memory exhausted ', what, '; MOZYME continues on the CPU'
#endif
  end subroutine acc_mozyme_give_up

  subroutine acc_mozyme_release()
! Frees the device buffers owned by this module (fd, pd, pacc).
    use iso_c_binding, only : c_associated
#ifdef _OPENACC
    if (c_associated(fd_c)) call c_acc_free(fd_c)
    if (c_associated(pd_c)) call c_acc_free(pd_c)
    if (c_associated(pacc_c)) call c_acc_free(pacc_c)
    fd_c = c_null_ptr
    pd_c = c_null_ptr
    pacc_c = c_null_ptr
    dev_len = 0
    pacc_len = 0
    fd_ok = .false.
    pd_ok = .false.
#endif
  end subroutine acc_mozyme_release

  integer(8) function acc_thread_slots(devnum)
! Number of SMs x maximum resident threads per SM of CUDA device devnum, from the driver API
! (libcuda, loaded with dlopen so that no link dependency is added), or 0 if unavailable.
    use iso_c_binding, only : c_associated, c_f_procpointer, c_null_char
    integer, intent(in) :: devnum
    type(c_ptr) :: h
    type(c_funptr) :: fp1, fp2
    procedure(cu_device_get), pointer :: get_dev
    procedure(cu_device_get_attribute), pointer :: get_attr
    integer(c_int) :: dev, nsm, nthr
    acc_thread_slots = 0
#ifdef _OPENACC
    h = c_dlopen('libcuda.so.1'//c_null_char, 2_c_int)        ! RTLD_NOW
    if (.not. c_associated(h)) return
    fp1 = c_dlsym(h, 'cuDeviceGet'//c_null_char)
    fp2 = c_dlsym(h, 'cuDeviceGetAttribute'//c_null_char)
    if (.not. c_associated(fp1) .or. .not. c_associated(fp2)) return
    call c_f_procpointer(fp1, get_dev)
    call c_f_procpointer(fp2, get_attr)
    if (get_dev(dev, int(devnum, c_int)) /= 0) return
    if (get_attr(nsm, 16_c_int, dev) /= 0) return               ! CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
    if (get_attr(nthr, 39_c_int, dev) /= 0) return              ! CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
    acc_thread_slots = int(nsm, 8)*int(nthr, 8)
#endif
  end function acc_thread_slots

  integer(8) function acc_free_memory()
! Free device memory in bytes (OpenACC acc_get_property, acc_property_free_memory).
    acc_free_memory = 0
#ifdef _OPENACC
    acc_free_memory = int(c_acc_get_property(c_acc_get_device_num(4_c_int), 4_c_int, 2_c_int), 8)
#endif
  end function acc_free_memory

  integer(8) function acc_mem_limit()
! MOPAC_ACC_MEM_LIMIT in bytes (suffix KB, MB or GB allowed), or -1 if not set.
    character(len=32) :: env
    integer :: stat, l, ios
    integer(8) :: mult
    double precision :: x
    acc_mem_limit = -1
    call get_environment_variable("MOPAC_ACC_MEM_LIMIT", env, status=stat)
    if (stat /= 0) return
    env = adjustl(env)
    l = len_trim(env)
    mult = 1
    if (l > 2) then
      if (env(l-1:l) == 'GB' .or. env(l-1:l) == 'gb') then
        mult = 1024_8**3
        l = l - 2
      else if (env(l-1:l) == 'MB' .or. env(l-1:l) == 'mb') then
        mult = 1024_8**2
        l = l - 2
      else if (env(l-1:l) == 'KB' .or. env(l-1:l) == 'kb') then
        mult = 1024_8
        l = l - 2
      end if
    end if
    read (env(1:l), *, iostat=ios) x
    if (ios == 0) acc_mem_limit = int(x*dble(mult), 8)
  end function acc_mem_limit

  subroutine acc_mozyme_plan()
! Decides which of the requested GPU paths fit in device memory (see the module header).
    use molkst_C, only : numat, mpack
    use MOZYME_C, only : nijbo, iorbs, cocc, cvir, icocc, icvir, fmo_dim
    use common_arrays_C, only : nat
    use chanel_C, only : iw
    integer, external :: acc_fock_mode, acc_diagg_mode, acc_density_mode
    logical :: want_fock, want_diagg, want_dens, f, g, d, sdg, ddg
    integer(8) :: free, total, avail, limit, headroom, b_nij, b_fp, b_pacc, b_lists, b_cand, b_lmo_g, b_lmo_d, slots
    integer(8) :: npair, npair_d, npair_sd, npair_dd, lcocc, lcvir, licocc, licvir
    integer :: i, j, dev, k
    double precision, parameter :: gb = 1024.d0**3
    character(len=64) :: enabled
    character(len=8) :: lev
    integer :: k0, stat
#ifdef _OPENACC
    want_fock = acc_fock_mode() > 0
    want_diagg = acc_diagg_mode() > 0
    want_dens = acc_density_mode() > 0
    allow_fock = .false.
    allow_diagg = .false.
    allow_dens = .false.
    allow_sdgpu = .false.
    allow_ddgpu = .false.
    if (.not. (want_fock .or. want_diagg .or. want_dens)) return
    call acc_mozyme_release()
    dev = c_acc_get_device_num(4_c_int)
    total = int(c_acc_get_property(int(dev, c_int), 4_c_int, 1_c_int), 8)
    free = int(c_acc_get_property(int(dev, c_int), 4_c_int, 2_c_int), 8)
    slots = acc_thread_slots(dev)
    if (slots == 0) slots = 200_8*2048_8          ! unknown: assume a very large device (conservative)
    avail = free + held_lists + held_cand      ! the stack reservation already held is credited in need()
    limit = acc_mem_limit()
    if (limit >= 0) avail = min(avail, limit)
    headroom = 256_8*1024_8**2 + total/20
! Sizes, in bytes
    npair = 0
    npair_sd = 0
    npair_dd = 0
    do i = 2, numat
      if (iorbs(i) == 0) cycle
      do j = 1, i - 1
        if (iorbs(j) == 0 .or. nijbo(i, j) < 0) cycle
        npair = npair + 1
        select case (acc_pair_class(nat(i), iorbs(i), nat(j), iorbs(j)))
        case (1)
          npair_sd = npair_sd + 1
        case (2)
          npair_dd = npair_dd + 1
        end select
      end do
    end do
    npair_d = npair_sd + npair_dd
    lcocc = 0; lcvir = 0; licocc = 0; licvir = 0
    if (allocated(cocc)) lcocc = size(cocc, kind=8)
    if (allocated(cvir)) lcvir = size(cvir, kind=8)
    if (allocated(icocc)) licocc = size(icocc, kind=8)
    if (allocated(icvir)) licvir = size(icvir, kind=8)
    b_nij = 4_8*int(numat, 8)**2
    b_fp = 16_8*int(mpack, 8)
    b_pacc = 8_8*int(mpack, 8)
    b_lists = 8_8*npair + 760_8*npair_d + 16_8*numat
    b_cand = 30_8*max(1000000_8, 2_8*int(fmo_dim, 8))     ! candidate buffer (20 bytes each) plus growth
    b_lmo_g = 16_8*(lcocc + lcvir) + 24_8*(licocc + licvir)
    b_lmo_d = 8_8*lcocc + 8_8*licocc
    lmem_sp = stack_rotate_sp*slots
    lmem_sd = stack_rotate_spd*slots
    lmem_dd = stack_rotate*slots
! Levels (module header): 1 everything; 2 d,d pairs on the CPU; 3 all d pairs on the CPU; 4 no
! density; 5 Fock only; 6 nothing. Level 1 is chosen here only if the d,d reservation is already
! held; otherwise it is decided after the first Fock builds (acc_dd_decide).
! MOPAC_ACC_MEM_LEVEL (testing) starts the search at the given level (1 forces the d,d kernel).
    k0 = 2
    if (npair_dd > 0 .and. lmem_held >= lmem_dd) k0 = 1
    call get_environment_variable("MOPAC_ACC_MEM_LEVEL", lev, status=stat)
    if (stat == 0) then
      read (lev, *, iostat=stat) k0
      if (stat /= 0 .or. k0 < 1 .or. k0 > 6) k0 = 1
    end if
    do k = k0, 6
      f = want_fock .and. k <= 5
      ddg = f .and. npair_dd > 0 .and. k == 1
      sdg = f .and. npair_sd > 0 .and. k <= 2
      d = want_dens .and. k <= 3
      g = want_diagg .and. k <= 4
      if (need() <= avail - headroom) exit
    end do
    allow_fock = f
    allow_diagg = g
    allow_dens = d
    allow_sdgpu = sdg
    allow_ddgpu = ddg
    plan_headroom = headroom
    dd_pending = k == 2 .and. k0 == 2 .and. stat /= 0 .and. npair_dd > 0 .and. allow_sdgpu
    t_dd_cpu = -1.d0
    t_sd_gpu = -1.d0
    t_iter = -1.d0
    t_last_build = -1
    enabled = ''
    if (allow_fock) then
      if ((allow_sdgpu .or. npair_sd == 0) .and. (allow_ddgpu .or. npair_dd == 0)) then
        enabled = trim(enabled)//' Fock'
      else if (allow_sdgpu .or. npair_sd == 0) then
        enabled = trim(enabled)//' Fock(d,d pairs on CPU)'
      else
        enabled = trim(enabled)//' Fock(d pairs on CPU)'
      end if
    end if
    if (allow_diagg) enabled = trim(enabled)//' DIAGG1'
    if (allow_dens) enabled = trim(enabled)//' density'
    if (len_trim(enabled) == 0) enabled = ' none (all on the CPU)'
    write (iw, '(a,f7.3,a,f7.3,a)', advance='no') ' ACC MOZYME: device memory free ', dble(free)/gb, ' of ', &
      dble(total)/gb, ' GB'
    if (limit >= 0) write (iw, '(a,f7.3,a)', advance='no') ' (MOPAC_ACC_MEM_LIMIT ', dble(limit)/gb, ' GB)'
    if (stat == 0) write (iw, '(a,i0,a)', advance='no') ' (MOPAC_ACC_MEM_LEVEL ', k0, ')'
    write (iw, '(a,f7.3,a,f7.3,a,f7.3,a,f7.3,a,f7.3,a,f7.3,a,f7.3,a,i0,a)') '; needed (GB): Fock ', &
      dble(b_nij + b_fp + b_lists + lmem_sp)/gb, ', +s,p-d pairs ', dble(lmem_sd - lmem_sp)/gb, ', +d,d pairs ', &
      dble(lmem_dd - lmem_sd)/gb, ', +DIAGG1 ', dble(b_cand + b_lmo_g)/gb, ', +density ', dble(b_pacc + b_lmo_d)/gb, &
      ', headroom ', dble(headroom)/gb, ', available ', dble(avail)/gb, ' (', slots, ' thread slots)'
    write (iw, '(a,i0,a,i0,a,i0)', advance='no') ' ACC MOZYME: NDDO pairs ', npair, ', of which s,p-d ', npair_sd, &
      ', d,d ', npair_dd
    if (dd_pending) write (iw, '(a)', advance='no') &
      ' (d,d pairs on the CPU; moved to the GPU after the first Fock builds if that pays)'
    write (iw, '(a)') ''
    write (iw, '(a,i0,2a)') ' ACC MOZYME: level ', k, '; GPU paths enabled:', trim(enabled)
    flush (iw)
#endif
  contains
    integer(8) function need()
      need = 0
      if (f .or. g .or. d) need = b_nij + b_fp
      if (f) need = need + b_lists + lmem() - min(lmem_held, lmem())
      if (g) need = need + b_cand + b_lmo_g
      if (d) need = need + b_pacc + b_lmo_d
    end function need
    integer(8) function lmem()
! Local-memory reservation of the largest pair kernel the Fock build will launch.
      lmem = lmem_sp
      if (sdg) lmem = max(lmem, lmem_sd)
      if (ddg) lmem = max(lmem, lmem_dd)
    end function lmem
  end subroutine acc_mozyme_plan

  subroutine acc_dd_decide(nsd, ndd)
! Level 2 by policy: once the CPU d,d pass, an SCF iteration and the GPU s,p-d kernel have been timed,
! move the d,d pairs to the GPU if that saves more than f_dd of an iteration and the reservation fits
! (module header). Called by acc_fz2n after a Fock build; takes effect from the next one.
    use chanel_C, only : iw
    integer, intent(in) :: nsd, ndd
    double precision :: est, saving
    integer(8) :: free, need
    if (t_dd_cpu < 0.d0 .or. t_iter <= 0.d0) return        ! not measured yet
    if (nsd > 0 .and. t_sd_gpu < 0.d0) return
    dd_pending = .false.
    est = 0.d0
    if (nsd > 0) est = dble(ndd)*(t_sd_gpu/dble(nsd))*k_dd
    saving = t_dd_cpu - est
    need = max(0_8, lmem_dd - lmem_held)
    free = acc_free_memory()
    write (iw, '(a,f8.4,a,f8.4,a,f8.3,a)', advance='no') ' ACC MOZYME: d,d pairs: CPU ', t_dd_cpu, &
      ' s per Fock build, GPU (estimated) ', est, ' s, SCF iteration ', t_iter, ' s'
    if (saving <= f_dd*t_iter) then
      write (iw, '(a)') '; stay on the CPU (saving under 5 % of an iteration)'
    else if (free - need < plan_headroom) then
      write (iw, '(a,f6.2,a,f6.2,a)') '; would pay, but the kernel reservation (', dble(need)/1024.d0**3, &
        ' GB) does not fit (free ', dble(free)/1024.d0**3, ' GB): stay on the CPU'
    else
      allow_ddgpu = .true.
      write (iw, '(a,f6.2,a)') '; moved to the GPU (reservation ', dble(need)/1024.d0**3, ' GB)'
    end if
    flush (iw)
  end subroutine acc_dd_decide

  subroutine acc_lmem_begin(predicted, free0)
! Before a pair kernel's launch: if it will raise the local-memory reservation (predicted, bytes,
! larger than what earlier kernels triggered), record the free memory so that acc_lmem_end can
! measure the increase; otherwise free0 = -1.
    integer(8), intent(in) :: predicted
    integer(8), intent(out) :: free0
    free0 = -1
    if (lmem_held < predicted) free0 = acc_free_memory()
  end subroutine acc_lmem_begin

  subroutine acc_lmem_end(predicted, free0, what)
! After the launch: report the measured increase of the reservation against the prediction, with a
! warning if they differ by more than 10 % (then the stack constant for that kernel is out of date).
    use chanel_C, only : iw
    integer(8), intent(in) :: predicted, free0
    character(len=*), intent(in) :: what
    integer(8) :: drop, expect
    if (free0 < 0) return
    drop = free0 - acc_free_memory()
    expect = predicted - lmem_held
    write (iw, '(3a,f6.2,a,f6.2,a)') ' ACC MOZYME: ', what, ' kernel stack reservation: predicted ', &
      dble(expect)/1024.d0**3, ' GB, measured ', dble(drop)/1024.d0**3, ' GB'
    if (abs(dble(drop) - dble(expect)) > 0.1d0*dble(expect)) write (iw, '(3a)') &
      ' ACC MOZYME: WARNING: the stack reservation differs from the prediction by more than 10 %; update the ', &
      what, ' stack constant'
    lmem_held = predicted
  end subroutine acc_lmem_end

  logical function acc_matches_device(dev, a, n, rtol)
! Compares 32 elements of a, spread over 1..n, with the device buffer dev, to relative tolerance
! rtol (0: bitwise). A cheap guard against a host write that bypassed the validity flags; it
! cannot prove equality.
    use iso_c_binding, only : c_loc, c_ptr
    type(c_ptr), intent(in) :: dev
    integer, intent(in) :: n
    double precision, intent(in), target :: a(n)
    double precision, intent(in) :: rtol
    double precision, target :: x
    integer :: k, i
    acc_matches_device = .true.
#ifdef _OPENACC
    do k = 0, 31
      i = 1 + int((int(n - 1, 8)*k)/31)
      call c_acc_memcpy_from_device(c_loc(x), dev_offset(dev, i), 8_c_long_long)
      if (abs(x - a(i)) > rtol*abs(a(i)) .or. (rtol == 0.d0 .and. x /= a(i))) then
        acc_matches_device = .false.
        return
      end if
    end do
#endif
  end function acc_matches_device

  type(c_ptr) function dev_offset(dev, i)
! Address of element i (1-based) of a double precision device buffer.
    use iso_c_binding, only : c_intptr_t
    type(c_ptr), intent(in) :: dev
    integer, intent(in) :: i
    integer(c_intptr_t) :: addr
    addr = transfer(dev, addr) + int(i - 1, c_intptr_t)*8_c_intptr_t
    dev_offset = transfer(addr, dev_offset)
  end function dev_offset

  subroutine acc_to_device(dev, a, n)
    use iso_c_binding, only : c_loc
    type(c_ptr), intent(in) :: dev
    integer, intent(in) :: n
    double precision, intent(in), target :: a(n)
#ifdef _OPENACC
    call c_acc_memcpy_to_device(dev, c_loc(a), int(n, c_long_long)*8_c_long_long)
#endif
  end subroutine acc_to_device

  subroutine acc_from_device(a, dev, n)
    use iso_c_binding, only : c_loc
    type(c_ptr), intent(in) :: dev
    integer, intent(in) :: n
    double precision, intent(inout), target :: a(n)
#ifdef _OPENACC
    call c_acc_memcpy_from_device(c_loc(a), dev, int(n, c_long_long)*8_c_long_long)
#endif
  end subroutine acc_from_device

  logical function acc_mozyme_sync()
! Returns .true. if nij was (re)built on this call. Consumers that keep data derived from nij must
! compare nij_gen with the generation they built for, since another consumer may have synced first.
#ifdef _OPENACC
    use molkst_C, only : numat, mpack, numcal
    use MOZYME_C, only : nijbo
#endif
    acc_mozyme_sync = .false.
#ifdef _OPENACC
    if (numcal == key_numcal .and. mpack == key_mpack .and. numat == key_numat) return
    if (allocated(nij)) then
      !$acc exit data delete(nij)
      deallocate (nij)
    end if
    call acc_mozyme_plan()
    call acc_update_nddo()
    allocate (nij(numat, numat))
    nij = nijbo(1:numat, 1:numat)
    if (allow_fock .or. allow_diagg .or. allow_dens) then
      !$acc enter data copyin(nij)
    end if
    key_numcal = numcal
    key_mpack = mpack
    key_numat = numat
    nij_gen = nij_gen + 1
    acc_mozyme_sync = .true.
#endif
  end function acc_mozyme_sync
end module acc_mozyme_C

integer function acc_fock_mode()
  implicit none
  integer, save :: mode = -1
#ifdef _OPENACC
  character(len=16) :: env
  integer :: stat
  if (mode < 0) then
    call get_environment_variable("MOPAC_ACC_FOCK", env, status=stat)
    if (stat /= 0) then
      mode = 0
    else if (trim(env) == 'test') then
      mode = 2
    else
      mode = 1
    end if
  end if
#else
  mode = 0
#endif
  acc_fock_mode = mode
end function acc_fock_mode

subroutine acc_fz2n(f, ptot, qe, iorbs, nat, coord, resident, p_on_device, prepare, supported)
! Adds the GPU part of the two-center Fock terms to f. f and ptot are copied to the device buffers
! fd and pd, the kernel adds to fd, and fd is copied back to f. If resident, f and ptot are the
! complete common_arrays f and p, so fd and pd remain valid for diagg1 (fd_ok, pd_ok).
! If p_on_device, pd already holds ptot (left by the GPU density build) and is not copied again,
! after a sampled comparison.
! prepare: only (re)build the pair lists and return supported = .false. if some NDDO pair is of a
! kind the GPU code does not handle (then fz2n uses the CPU for everything).
#ifdef _OPENACC
  use molkst_C, only : numat, mpack
  use parameters_C, only : dorbs, iod
  use acc_mozyme_C, only : nij, nij_gen, acc_mozyme_sync, acc_mozyme_buffers, acc_to_device, acc_from_device, &
    fd_c, pd_c, fd_ok, pd_ok, acc_matches_device, allow_fock, held_lists, allow_sdgpu, allow_ddgpu, n_sdpairs, &
    n_ddpairs, lmem_sp, lmem_sd, lmem_dd, lmem_held, acc_pair_class, acc_lmem_begin, acc_lmem_end, &
    dd_pending, t_iter, t_last_build, t_sd_gpu, acc_dd_decide, &
    nd => dp_nd, nsd => dp_nsd, ndd => dp_ndd, dpi => dp_i, dpj => dp_j, dcls => dp_cls, isd => dp_isd, &
    idd => dp_idd, dstart => dp_start, dlist => dp_list, cdi => dp_ci, cdj => dp_cj
  use iso_c_binding, only : c_f_pointer
  use common_arrays_C, only : fcom => f, pcom => p
  use chanel_C, only : iw
#endif
  implicit none
  double precision, intent(inout) :: f(*)
  double precision, intent(in) :: ptot(*), qe(*), coord(3,*)
  integer, intent(in) :: iorbs(*), nat(*)
  logical, intent(in) :: resident, p_on_device, prepare
  logical, intent(out) :: supported
#ifdef _OPENACC
  logical :: p_reuse
  integer, save :: nreuse = 0
  integer(8) :: free0, tc0, tc1, trate
  integer :: c
  double precision, pointer, contiguous :: fdev(:), pdev(:)
  integer, save :: nnear = 0, my_gen = -1
  integer, allocatable, save :: nstart(:), near(:)
! The NDDO pairs with d orbitals (dpi, dpj, ... ) are module arrays of acc_mozyme_C (dp_*).
  logical, save :: unsupported = .false.
  integer :: kd, m
  integer, allocatable :: dtmp(:)
  logical :: spi, spj, rebuilt
  integer :: a, b, ii, jj, k
  supported = .false.
  rebuilt = acc_mozyme_sync()   ! separate statement: Fortran need not evaluate a function inside .or.
  if ((rebuilt .or. my_gen /= nij_gen .or. .not. allow_fock) .and. allocated(nstart)) then
    !$acc exit data delete(nstart, near, dpi, dpj, dcls, isd, idd, dstart, dlist, cdi, cdj)
    deallocate (nstart, near, dpi, dpj, dcls, isd, idd, dstart, dlist, cdi, cdj)
    held_lists = 0
    my_gen = -1
  end if
  if (.not. allow_fock) return            ! not enabled, or not enough device memory: CPU
  if (my_gen /= nij_gen) then
! New geometry or new set of interactions: rebuild the NDDO s,p neighbor lists (CSR).
    allocate (nstart(numat + 1))
    nnear = 0
    do k = 1, 2
      if (k == 2) allocate (near(max(1, nnear)))
      nnear = 0
      do a = 1, numat
        if (k == 2) nstart(a) = nnear + 1
        spi = (iorbs(a) == 1 .or. iorbs(a) == 4) .and. .not. dorbs(nat(a)) .and. iod(nat(a)) == 0
        if (.not. spi) cycle
        do b = 1, numat
          if (b == a) cycle
          spj = (iorbs(b) == 1 .or. iorbs(b) == 4) .and. .not. dorbs(nat(b)) .and. iod(nat(b)) == 0
          if (.not. spj) cycle
          ii = max(a, b)
          jj = min(a, b)
          if (nij(ii, jj) < 0) cycle
          nnear = nnear + 1
          if (k == 2) near(nnear) = b
        end do
      end do
    end do
    nstart(numat + 1) = nnear + 1
! NDDO pairs with d orbitals (the complement of the s,p list among NDDO pairs)
    unsupported = .false.
    nd = 0
    do k = 1, 2
      if (k == 2) allocate (dpi(max(1, nd)), dpj(max(1, nd)), dcls(max(1, nd)))
      nd = 0
      do a = 2, numat
        if (iorbs(a) == 0) cycle
        do b = 1, a - 1
          if (iorbs(b) == 0 .or. nij(a, b) < 0) cycle
          c = acc_pair_class(nat(a), iorbs(a), nat(b), iorbs(b))
          if (c == 0) cycle
          if (iorbs(a) <= 5 .and. iorbs(b) <= 5) unsupported = .true.   ! not an s,p pair and not a d pair
          nd = nd + 1
          if (k == 2) then
            dpi(nd) = a
            dpj(nd) = b
            dcls(nd) = c
          end if
        end do
      end do
    end do
    nsd = count(dcls(1:nd) == 1)
    ndd = nd - nsd
    allocate (isd(max(1, nsd)), idd(max(1, ndd)))
    nsd = 0
    ndd = 0
    do kd = 1, nd
      if (dcls(kd) == 1) then
        nsd = nsd + 1
        isd(nsd) = kd
      else
        ndd = ndd + 1
        idd(ndd) = kd
      end if
    end do
    allocate (dstart(numat + 1), dlist(max(1, 2*nd)), cdi(45, max(1, nd)), cdj(45, max(1, nd)))
    dstart = 0
    do kd = 1, nd
      dstart(dpi(kd)) = dstart(dpi(kd)) + 1
      dstart(dpj(kd)) = dstart(dpj(kd)) + 1
    end do
    m = 1
    do a = 1, numat
      k = dstart(a)
      dstart(a) = m
      m = m + k
    end do
    dstart(numat + 1) = m
    allocate (dtmp(numat))
    dtmp(1:numat) = dstart(1:numat)
    do kd = 1, nd
      dlist(dtmp(dpi(kd))) = kd
      dtmp(dpi(kd)) = dtmp(dpi(kd)) + 1
      dlist(dtmp(dpj(kd))) = -kd
      dtmp(dpj(kd)) = dtmp(dpj(kd)) + 1
    end do
    deallocate (dtmp)
    if (unsupported) write (iw, '(a)') ' ACC FOCK: NDDO pairs of an unsupported kind; the GPU Fock build is not used'
    !$acc enter data copyin(nstart, near, dpi, dpj, dcls, isd, idd, dstart, dlist) create(cdi, cdj)
    n_sdpairs = nsd
    n_ddpairs = ndd
    held_lists = 4_8*(size(nstart, kind=8) + size(near, kind=8) + size(dpi, kind=8) + size(dpj, kind=8) + &
      size(dcls, kind=8) + size(isd, kind=8) + size(idd, kind=8) + size(dstart, kind=8) + size(dlist, kind=8)) + &
      8_8*(size(cdi, kind=8) + size(cdj, kind=8))
    my_gen = nij_gen
  end if
  if (unsupported) return
! The device buffers are allocated here, in the prepare call, so that a failure is known before
! fz2n decides to skip its CPU pair loop.
  if (.not. acc_mozyme_buffers(mpack)) return
  supported = .true.
  if (prepare) return
  call system_clock(tc0, trate)
  if (t_last_build >= 0) t_iter = dble(tc0 - t_last_build)/dble(trate)   ! one SCF iteration
  t_last_build = tc0
  call acc_to_device(fd_c, f, mpack)
  p_reuse = .false.
  if (p_on_device .and. allocated(pcom)) then
    if (loc(ptot(1)) == loc(pcom(1))) p_reuse = acc_matches_device(pd_c, ptot, mpack, 0.d0)
    if (.not. p_reuse) write (iw, '(a)') ' ACC FOCK: device copy of P was stale; copied again'
  end if
  if (p_reuse) then
    nreuse = nreuse + 1
    if (nreuse == 1) write (iw, '(a)') ' ACC FOCK: using the device copy of P from the GPU density build'
  else
    call acc_to_device(pd_c, ptot, mpack)
  end if
  call c_f_pointer(fd_c, fdev, [mpack])
  call c_f_pointer(pd_c, pdev, [mpack])
  call acc_fz2n_kernel(numat, mpack, fdev, pdev, qe, iorbs, nat, coord, nij, nstart, near, max(1, nnear))
  lmem_held = max(lmem_held, lmem_sp)
! Pairs with d orbitals: each kernel's first launch measures the local-memory reservation it triggers.
  if (nsd > 0 .and. allow_sdgpu) then
    call acc_lmem_begin(lmem_sd, free0)
    call system_clock(tc0)
    call acc_fz2n_sdpairs(numat, mpack, nd, nsd, fdev, pdev, iorbs, nat, coord, nij, dpi, dpj, isd, cdi, cdj)
    call system_clock(tc1)
    if (free0 < 0) t_sd_gpu = dble(tc1 - tc0)/dble(trate)   ! not the launch that set up the reservation
    call acc_lmem_end(lmem_sd, free0, 's,p-d pair (rotate_spd)')
  end if
  if (ndd > 0 .and. allow_ddgpu) then
    call acc_lmem_begin(lmem_dd, free0)
    call acc_fz2n_ddpairs(numat, mpack, nd, ndd, fdev, pdev, iorbs, nat, coord, nij, dpi, dpj, idd, cdi, cdj)
    call acc_lmem_end(lmem_dd, free0, 'd,d pair (rotate)')
  end if
  if ((nsd > 0 .and. allow_sdgpu) .or. (ndd > 0 .and. allow_ddgpu)) call acc_fz2n_dsum(numat, mpack, nd, fdev, &
    iorbs, nij, dstart, dlist, dcls, allow_sdgpu, allow_ddgpu, cdi, cdj)
  call acc_from_device(f, fd_c, mpack)
  if (dd_pending) call acc_dd_decide(nsd, ndd)
  fd_ok = .false.
  pd_ok = .false.
  if (resident .and. allocated(fcom) .and. allocated(pcom)) then
    fd_ok = (loc(f(1)) == loc(fcom(1)))
    pd_ok = (loc(ptot(1)) == loc(pcom(1)))
  end if
#else
  supported = .false.
#endif
end subroutine acc_fz2n

#ifdef _OPENACC
subroutine acc_fz2n_sdpairs(numat, mpack, nd, nsd, f, ptot, iorbs, nat, coord, nij, dpi, dpj, isd, cdi, cdj)
! NDDO pairs of one s,p,d atom with an s or s,p atom, one thread per pair: integrals from rotate_spd
! (stack sized for this case), then the focd2z contraction (acc_dpair_contract).
  implicit none
  integer, intent(in) :: numat, mpack, nd, nsd
  double precision, intent(inout) :: f(mpack)
  double precision, intent(in) :: ptot(mpack), coord(3, numat)
  integer, intent(in) :: iorbs(numat), nat(numat), nij(numat, numat), dpi(nd), dpj(nd), isd(nsd)
  double precision, intent(inout) :: cdi(45, nd), cdj(45, nd)
  integer :: m, kd, ii, jj
  double precision :: w(450), e1b(45), e2a(45), enuc
  !$acc routine(rotate_spd) seq
  !$acc routine(acc_dpair_contract) seq
  !$acc data present(nij, dpi, dpj, isd, cdi, cdj) copyin(iorbs, nat, coord) deviceptr(f, ptot)
  !$acc parallel loop gang vector_length(64) private(w, e1b, e2a, enuc, kd, ii, jj)
  do m = 1, nsd
    kd = isd(m)
    ii = dpi(kd)
    jj = dpj(kd)
    call rotate_spd(nat(ii), nat(jj), coord(1, ii), coord(1, jj), w, e1b, e2a, enuc)
    call acc_dpair_contract(kd, nd, mpack, iorbs(ii), iorbs(jj), nij(ii, ii), nij(jj, jj), nij(ii, jj), w, f, &
      ptot, cdi, cdj)
  end do
  !$acc end data
end subroutine acc_fz2n_sdpairs

subroutine acc_fz2n_ddpairs(numat, mpack, nd, ndd, f, ptot, iorbs, nat, coord, nij, dpi, dpj, idd, cdi, cdj)
! The other NDDO pairs with d orbitals (two s,p,d atoms), one thread per pair: integrals from the
! general rotate, whose per-thread stack is about 71 KB, then the focd2z contraction.
  implicit none
  integer, intent(in) :: numat, mpack, nd, ndd
  double precision, intent(inout) :: f(mpack)
  double precision, intent(in) :: ptot(mpack), coord(3, numat)
  integer, intent(in) :: iorbs(numat), nat(numat), nij(numat, numat), dpi(nd), dpj(nd), idd(ndd)
  double precision, intent(inout) :: cdi(45, nd), cdj(45, nd)
  integer :: m, kd, ii, jj, kr
  double precision :: w(2025), e1b(45), e2a(45), enuc
  !$acc routine(rotate) seq
  !$acc routine(acc_dpair_contract) seq
  !$acc data present(nij, dpi, dpj, idd, cdi, cdj) copyin(iorbs, nat, coord) deviceptr(f, ptot)
  !$acc parallel loop gang vector_length(64) private(w, e1b, e2a, enuc, kd, ii, jj, kr)
  do m = 1, ndd
    kd = idd(m)
    ii = dpi(kd)
    jj = dpj(kd)
    kr = 1
    call rotate(nat(ii), nat(jj), coord(1, ii), coord(1, jj), w, kr, e1b, e2a, enuc)
    call acc_dpair_contract(kd, nd, mpack, iorbs(ii), iorbs(jj), nij(ii, ii), nij(jj, jj), nij(ii, jj), w, f, &
      ptot, cdi, cdj)
  end do
  !$acc end data
end subroutine acc_fz2n_ddpairs

subroutine acc_dpair_contract(kd, nd, mpack, iab, jba, di, dj, l, w, f, ptot, cdi, cdj)
! The contraction of focd2z (direct mode, wk = wj) for NDDO pair kd with integrals w: subtracts the
! exchange terms from the pair's off-diagonal block of f (one writer per block) and stores the
! contributions to the two diagonal blocks in cdi(:,kd) (higher atom) and cdj(:,kd) (lower atom);
! acc_fz2n_dsum adds those to f.
  implicit none
  !$acc routine seq
  integer, intent(in) :: kd, nd, mpack, iab, jba, di, dj, l
  double precision, intent(in) :: w(*), ptot(mpack)
  double precision, intent(inout) :: f(mpack), cdi(45, nd), cdj(45, nd)
  integer :: i, j, k, ll, ka, kc, ij, kl, loop, ik, il, jk, jl, x
  double precision :: aa, bb, wv
  do x = 1, 45
    cdi(x, kd) = 0.d0
    cdj(x, kd) = 0.d0
  end do
  loop = 0
  do i = 1, iab
    ka = (i*(i - 1))/2
    aa = 2.0d0
    do j = 1, i
      if (i == j) aa = 1.0d0
      ij = ka + j
      do k = 1, jba
        kc = (k*(k - 1))/2
        bb = 2.0d0
        do ll = 1, k
          if (k == ll) bb = 1.0d0
          kl = kc + ll
          loop = loop + 1
          wv = w(loop)
          cdi(ij, kd) = cdi(ij, kd) + bb*wv*ptot(dj + kl)
          cdj(kl, kd) = cdj(kl, kd) + aa*wv*ptot(di + ij)
          wv = w(loop)*aa*bb*0.125d0
          ik = (i - 1)*jba + k
          il = (i - 1)*jba + ll
          jk = (j - 1)*jba + k
          jl = (j - 1)*jba + ll
          f(l + ik) = f(l + ik) - wv*ptot(l + jl)
          f(l + il) = f(l + il) - wv*ptot(l + jk)
          f(l + jk) = f(l + jk) - wv*ptot(l + il)
          f(l + jl) = f(l + jl) - wv*ptot(l + ik)
        end do
      end do
    end do
  end do
end subroutine acc_dpair_contract

subroutine acc_fz2n_dsum(numat, mpack, nd, f, iorbs, nij, dstart, dlist, dcls, sd_on, dd_on, cdi, cdj)
! Adds the diagonal-block contributions of the pairs with d orbitals computed on the GPU (kinds
! enabled by sd_on, dd_on) to f, one gang per atom, summing over the atom's pairs in a fixed order.
  implicit none
  integer, intent(in) :: numat, mpack, nd
  double precision, intent(inout) :: f(mpack)
  integer, intent(in) :: iorbs(numat), nij(numat, numat), dstart(numat + 1), dlist(*), dcls(nd)
  logical, intent(in) :: sd_on, dd_on
  double precision, intent(in) :: cdi(45, nd), cdj(45, nd)
  integer :: a, e, npk, m, kd
  double precision :: s
  !$acc data present(nij, dstart, dlist, dcls, cdi, cdj) copyin(iorbs) deviceptr(f)
  !$acc parallel loop gang vector_length(64) private(npk)
  do a = 1, numat
    if (dstart(a + 1) == dstart(a)) cycle
    npk = (iorbs(a)*(iorbs(a) + 1))/2
    !$acc loop vector private(s, m, kd)
    do e = 1, npk
      s = 0.d0
      do m = dstart(a), dstart(a + 1) - 1
        kd = abs(dlist(m))
        if (dcls(kd) == 1) then
          if (.not. sd_on) cycle
        else
          if (.not. dd_on) cycle
        end if
        if (dlist(m) > 0) then
          s = s + cdi(e, kd)
        else
          s = s + cdj(e, kd)
        end if
      end do
      f(nij(a, a) + e) = f(nij(a, a) + e) + s
    end do
  end do
  !$acc end data
end subroutine acc_fz2n_dsum
#endif

subroutine acc_fz2n_cpu_dpairs(f, ptot, iorbs, nat, coord)
! The NDDO pairs with d orbitals that the device-memory plan keeps on the CPU (acc_pair_on_gpu false),
! for fz2n, with OpenMP: the scheme of the GPU kernels (integrals, acc_dpair_contract, then the
! diagonal blocks summed per atom over its pairs in a fixed order), so the result does not depend on
! the number of threads. Uses the pair lists built by the preceding acc_fz2n (prepare) call.
#ifdef _OPENACC
  use molkst_C, only : numat, mpack
  use acc_mozyme_C, only : nij, allow_sdgpu, allow_ddgpu, t_dd_cpu, nd => dp_nd, dpi => dp_i, dpj => dp_j, dcls => dp_cls, &
    dstart => dp_start, dlist => dp_list, cdi => dp_ci, cdj => dp_cj
#endif
  implicit none
  double precision, intent(inout) :: f(*)
  double precision, intent(in) :: ptot(*), coord(3, *)
  integer, intent(in) :: iorbs(*), nat(*)
#ifdef _OPENACC
  logical :: oncpu(2)
  integer :: kd, ii, jj, kr, a, e, npk, m
  integer(8) :: c0, c1, rate
  double precision :: w(2025), e1b(45), e2a(45), enuc, s
  oncpu(1) = .not. allow_sdgpu
  oncpu(2) = .not. allow_ddgpu
  if (nd == 0 .or. .not. (oncpu(1) .or. oncpu(2))) return
  call system_clock(c0, rate)
!$omp parallel do schedule(dynamic, 16) default(shared) private(kd, ii, jj, kr, w, e1b, e2a, enuc)
  do kd = 1, nd
    if (.not. oncpu(dcls(kd))) cycle
    ii = dpi(kd)
    jj = dpj(kd)
    if (dcls(kd) == 1) then
      call rotate_spd(nat(ii), nat(jj), coord(1, ii), coord(1, jj), w, e1b, e2a, enuc)
    else
      kr = 1
      call rotate(nat(ii), nat(jj), coord(1, ii), coord(1, jj), w, kr, e1b, e2a, enuc)
    end if
    call acc_dpair_contract(kd, nd, mpack, iorbs(ii), iorbs(jj), nij(ii, ii), nij(jj, jj), nij(ii, jj), w, f, &
      ptot, cdi, cdj)
  end do
!$omp end parallel do
!$omp parallel do schedule(dynamic, 64) default(shared) private(a, e, npk, m, kd, s)
  do a = 1, numat
    if (dstart(a + 1) == dstart(a)) cycle
    npk = (iorbs(a)*(iorbs(a) + 1))/2
    do e = 1, npk
      s = 0.d0
      do m = dstart(a), dstart(a + 1) - 1
        kd = abs(dlist(m))
        if (.not. oncpu(dcls(kd))) cycle
        if (dlist(m) > 0) then
          s = s + cdi(e, kd)
        else
          s = s + cdj(e, kd)
        end if
      end do
      f(nij(a, a) + e) = f(nij(a, a) + e) + s
    end do
  end do
!$omp end parallel do
  call system_clock(c1)
  t_dd_cpu = dble(c1 - c0)/dble(rate)     ! the d,d pass when only d,d pairs are on the CPU (level 2)
#endif
end subroutine acc_fz2n_cpu_dpairs
#ifdef _OPENACC

subroutine acc_fz2n_kernel(numat, mpack, f, ptot, qe, iorbs, nat, coord, nij, nstart, near, lnear)
  implicit none
  integer, intent(in) :: numat, mpack, lnear
  double precision, intent(inout) :: f(mpack)
  double precision, intent(in) :: ptot(mpack), qe(numat), coord(3, numat)
  integer, intent(in) :: iorbs(numat), nat(numat), nij(numat, numat), nstart(numat + 1), near(lnear)
  integer :: a, b, k, d, na
  double precision :: c(10), cm, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, sm
  !$acc routine(acc_fz_near_sp) seq
  !$acc routine(acc_fz_far) seq
  !$acc data copyin(qe, coord, iorbs, nat) present(nij, nstart, near) deviceptr(f, ptot)
  !$acc parallel loop gang vector_length(128) private(na, d, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, sm)
  do a = 1, numat
    na = iorbs(a)
    if (na == 0) cycle
    d = nij(a, a)
    s1 = 0.d0; s2 = 0.d0; s3 = 0.d0; s4 = 0.d0; s5 = 0.d0
    s6 = 0.d0; s7 = 0.d0; s8 = 0.d0; s9 = 0.d0; s10 = 0.d0; sm = 0.d0
! NDDO pairs, s,p atoms only (the list is empty for an atom with d orbitals)
    !$acc loop vector private(b, c) reduction(+:s1, s2, s3, s4, s5, s6, s7, s8, s9, s10)
    do k = nstart(a), nstart(a + 1) - 1
      b = near(k)
      call acc_fz_near_sp(a, b, numat, mpack, f, ptot, iorbs, nat, coord, nij, c)
      s1 = s1 + c(1); s2 = s2 + c(2); s3 = s3 + c(3); s4 = s4 + c(4); s5 = s5 + c(5)
      s6 = s6 + c(6); s7 = s7 + c(7); s8 = s8 + c(8); s9 = s9 + c(9); s10 = s10 + c(10)
    end do
! Point-charge pairs (monopole, and dipole where nijbo = -2)
    !$acc loop vector private(c, cm) reduction(+:s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, sm)
    do b = 1, numat
      if (b == a) cycle
      if (iorbs(b) == 0) cycle
      if (nij(max(a, b), min(a, b)) >= 0) cycle
      call acc_fz_far(a, b, nij(max(a, b), min(a, b)), numat, mpack, ptot, qe, iorbs, nat, coord, nij, c, cm)
      s1 = s1 + c(1); s2 = s2 + c(2); s3 = s3 + c(3); s4 = s4 + c(4); s5 = s5 + c(5)
      s6 = s6 + c(6); s7 = s7 + c(7); s8 = s8 + c(8); s9 = s9 + c(9); s10 = s10 + c(10)
      sm = sm + cm
    end do
    f(d + 1) = f(d + 1) + s1
    if (na > 1) then
      f(d + 2) = f(d + 2) + s2
      f(d + 3) = f(d + 3) + s3
      f(d + 4) = f(d + 4) + s4
      f(d + 5) = f(d + 5) + s5
      f(d + 6) = f(d + 6) + s6
      f(d + 7) = f(d + 7) + s7
      f(d + 8) = f(d + 8) + s8
      f(d + 9) = f(d + 9) + s9
      f(d + 10) = f(d + 10) + s10
    end if
    if (na > 4) then          ! monopole on the diagonal of the d orbitals
      f(d + 15) = f(d + 15) + sm
      f(d + 21) = f(d + 21) + sm
      f(d + 28) = f(d + 28) + sm
      f(d + 36) = f(d + 36) + sm
      f(d + 45) = f(d + 45) + sm
    end if
  end do
  !$acc end data
end subroutine acc_fz2n_kernel

subroutine acc_fz_near_sp(a, b, numat, mpack, f, ptot, iorbs, nat, coord, nij, c)
! Contribution of the NDDO pair (a, b) to the diagonal block of a, as packed lower triangle c(1:10);
! if a > b, also subtracts the exchange term from the off-diagonal block (a, b) of f.
  implicit none
  !$acc routine seq
  !$acc routine(rotate_sp) seq
  integer, intent(in) :: a, b, numat, mpack
  double precision, intent(inout) :: f(mpack)
  double precision, intent(in) :: ptot(mpack), coord(3, numat)
  integer, intent(in) :: iorbs(numat), nat(numat), nij(numat, numat)
  double precision, intent(out) :: c(10)
  integer, parameter :: jindx(10,16) = reshape([ &
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 31, 32, 33, 34, 35, 36, &
    & 37, 38, 39, 40, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 11, 12, 13, 14, 15, 16, 17, 18, 19, &
    & 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 71, 72, &
    & 73, 74, 75, 76, 77, 78, 79, 80, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, &
    & 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 81, 82, 83, 84, 85, 86, 87, 88, &
    & 89, 90, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, &
    & 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100], [10,16])
  integer, parameter :: jjndx(10,16) = reshape([ &
    1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 2, 12, 22, 32, 42, 52, 62, 72, 82, 92, 4, 14, 24, 34, &
    & 44, 54, 64, 74, 84, 94, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97, 2, 12, 22, 32, 42, 52, 62, 72, &
    & 82, 92, 3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 8, 18, &
    & 28, 38, 48, 58, 68, 78, 88, 98, 4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 5, 15, 25, 35, 45, 55, &
    & 65, 75, 85, 95, 6, 16, 26, 36, 46, 56, 66, 76, 86, 96, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, &
    & 7, 17, 27, 37, 47, 57, 67, 77, 87, 97, 8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 9, 19, 29, 39, &
    & 49, 59, 69, 79, 89, 99, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [10,16])
  integer, parameter :: kkind(16,16) = reshape([ &
    1, 2, 4, 7, 11, 12, 14, 17, 31, 32, 34, 37, 61, 62, 64, 67, 2, 3, 5, 8, 12, 13, 15, 18, 32, 33, &
    & 35, 38, 62, 63, 65, 68, 4, 5, 6, 9, 14, 15, 16, 19, 34, 35, 36, 39, 64, 65, 66, 69, 7, 8, 9, &
    & 10, 17, 18, 19, 20, 37, 38, 39, 40, 67, 68, 69, 70, 11, 12, 14, 17, 21, 22, 24, 27, 41, 42, &
    & 44, 47, 71, 72, 74, 77, 12, 13, 15, 18, 22, 23, 25, 28, 42, 43, 45, 48, 72, 73, 75, 78, 14, &
    & 15, 16, 19, 24, 25, 26, 29, 44, 45, 46, 49, 74, 75, 76, 79, 17, 18, 19, 20, 27, 28, 29, 30, &
    & 47, 48, 49, 50, 77, 78, 79, 80, 31, 32, 34, 37, 41, 42, 44, 47, 51, 52, 54, 57, 81, 82, 84, &
    & 87, 32, 33, 35, 38, 42, 43, 45, 48, 52, 53, 55, 58, 82, 83, 85, 88, 34, 35, 36, 39, 44, 45, &
    & 46, 49, 54, 55, 56, 59, 84, 85, 86, 89, 37, 38, 39, 40, 47, 48, 49, 50, 57, 58, 59, 60, 87, &
    & 88, 89, 90, 61, 62, 64, 67, 71, 72, 74, 77, 81, 82, 84, 87, 91, 92, 94, 97, 62, 63, 65, 68, &
    & 72, 73, 75, 78, 82, 83, 85, 88, 92, 93, 95, 98, 64, 65, 66, 69, 74, 75, 76, 79, 84, 85, 86, &
    & 89, 94, 95, 96, 99, 67, 68, 69, 70, 77, 78, 79, 80, 87, 88, 89, 90, 97, 98, 99, 100], [16,16])
  integer :: ii, jj, ia, ja, di, dj, l, i, j, k, m, jk, kj, li
  double precision :: w(100), e1b(10), e2a(10), enuc, p(16), sum, sumoff, sumdia
  ii = max(a, b)
  jj = min(a, b)
  call rotate_sp(nat(ii), nat(jj), coord(1, ii), coord(1, jj), w, e1b, e2a, enuc)
  ia = iorbs(ii)
  ja = iorbs(jj)
  di = nij(ii, ii)
  dj = nij(jj, jj)
  l = nij(ii, jj)
  do i = 1, 10
    c(i) = 0.d0
  end do
  if (ia == 4 .and. ja == 4) then
! Heavy atom - heavy atom: jab_for_MOZYME (Coulomb) and kab_for_MOZYME (exchange)
    if (a == ii) then
      m = 0
      do j = 1, 4
        do k = 1, 4
          m = m + 1
          jk = min(j, k)
          kj = k + j - jk
          p(m) = ptot(dj + (kj*(kj - 1))/2 + jk)
        end do
      end do
      do j = 1, 16
        do i = 1, 10
          c(i) = c(i) + p(j)*w(jjndx(i, j))
        end do
      end do
      do j = 1, 16
        sum = 0.d0
        do i = 1, 16
          sum = sum + ptot(l + i)*w(kkind(i, j))
        end do
        f(l + j) = f(l + j) - sum*0.5d0
      end do
    else
      m = 0
      do j = 1, 4
        do k = 1, 4
          m = m + 1
          jk = min(j, k)
          kj = k + j - jk
          p(m) = ptot(di + (kj*(kj - 1))/2 + jk)
        end do
      end do
      do j = 1, 16
        do i = 1, 10
          c(i) = c(i) + p(j)*w(jindx(i, j))
        end do
      end do
    end if
  else if (ia == 4 .or. ja == 4) then
! Heavy atom - light atom, either order. W runs over the heavy atom's 10 pairs.
    if ((a == ii .and. ia == 4) .or. (a == jj .and. ja == 4)) then
      if (ia == 4) then
        li = dj + 1
      else
        li = di + 1
      end if
      do k = 1, 10
        c(k) = ptot(li)*w(k)
      end do
    else
      if (ia == 4) then
        li = di
      else
        li = dj
      end if
      sumoff = 0.d0
      sumdia = 0.d0
      k = 0
      do i = 1, 4
        do j = 1, i - 1
          li = li + 1
          k = k + 1
          sumoff = sumoff + ptot(li)*w(k)
        end do
        li = li + 1
        k = k + 1
        sumdia = sumdia + ptot(li)*w(k)
      end do
      c(1) = sumoff*2.d0 + sumdia
    end if
    if (a == ii) then
      do i = 1, 4
        sum = 0.d0
        do j = 1, 4
          sum = sum + ptot(l + j)*w((max(i, j)*(max(i, j) - 1))/2 + min(i, j))
        end do
        f(l + i) = f(l + i) - sum*0.5d0
      end do
    end if
  else
! Light atom - light atom
    if (a == ii) then
      c(1) = ptot(dj + 1)*w(1)
      f(l + 1) = f(l + 1) - ptot(l + 1)*w(1)*0.5d0
    else
      c(1) = ptot(di + 1)*w(1)
    end if
  end if
end subroutine acc_fz_near_sp

subroutine acc_fz_far(a, b, code, numat, mpack, ptot, qe, iorbs, nat, coord, nij, c, cm)
! Contribution of the point-charge pair (a, b) to the diagonal block of a: c(1:10) for the s,p
! part (packed), cm for each d-orbital diagonal element. code = nijbo(max(a,b), min(a,b)) < 0.
  use parameters_C, only : am, ad, dd
  use funcon_C, only : ev, a0
  use molkst_C, only : l_feather
  implicit none
  !$acc routine seq
  !$acc routine(to_point) seq
  integer, intent(in) :: a, b, code, numat, mpack
  double precision, intent(in) :: ptot(mpack), qe(numat), coord(3, numat)
  integer, intent(in) :: iorbs(numat), nat(numat), nij(numat, numat)
  double precision, intent(out) :: c(10), cm
  integer :: ii, jj, ni, nj, i1, j1, i
  double precision :: dx, dy, dz, r2, r, rij, aee, w1, w2, w3, w4, w5, w6, w7, da, ade, rp, rm, ri2, ri5, &
    point, const, qa
  ii = max(a, b)
  jj = min(a, b)
  do i = 1, 10
    c(i) = 0.d0
  end do
  dx = coord(1, ii) - coord(1, jj)
  dy = coord(2, ii) - coord(2, jj)
  dz = coord(3, ii) - coord(3, jj)
  r2 = dx*dx + dy*dy + dz*dz
  ni = nat(ii)
  nj = nat(jj)
  aee = 0.5d0/am(ni) + 0.5d0/am(nj)
  if (l_feather) then
    rij = sqrt(r2)
    call to_point(rij, point, const)
    w1 = ev/sqrt(r2/(a0**2) + aee**2)
    w1 = w1*const + (1.d0 - const)*point
  else
    w1 = ev/sqrt(r2/(a0**2) + aee**2)
  end if
! Monopole: qe(b)*w1 on every diagonal element of a
  qa = qe(b)*w1
  c(1) = qa
  if (iorbs(a) > 1) then
    c(3) = qa
    c(6) = qa
    c(10) = qa
  end if
  cm = qa
  if (code /= -2) return
! Dipole
  i1 = nij(ii, ii)
  j1 = nij(jj, jj)
  w2 = 0.d0; w3 = 0.d0; w4 = 0.d0; w5 = 0.d0; w6 = 0.d0; w7 = 0.d0
  r = sqrt(r2)
  dx = dx/r
  dy = dy/r
  dz = dz/r
  if (abs(dz) > 0.99999999d0) dz = sign(1.d0, dz)
  if (iorbs(ii) > 1) then
    da = dd(ni)
    ade = 0.5d0/ad(ni) + 0.5d0/am(nj)
    rp = sqrt((r/a0 + da)**2 + ade**2)
    rm = sqrt((r/a0 - da)**2 + ade**2)
    ri2 = ev*(0.5d0/rp - 0.5d0/rm)
    if (l_feather) then
      call to_point(r, point, const)
      ri2 = ri2*const
    end if
    w5 = ri2*dx
    w6 = ri2*dy
    w7 = ri2*dz
  end if
  if (iorbs(jj) > 1) then
    da = dd(nj)
    ade = 0.5d0/am(ni) + 0.5d0/ad(nj)
    rp = sqrt((r/a0 + da)**2 + ade**2)
    rm = sqrt((r/a0 - da)**2 + ade**2)
    ri5 = -ev*(0.5d0/rp - 0.5d0/rm)
    if (l_feather) then
      call to_point(r, point, const)
      ri5 = ri5*const
    end if
    w2 = ri5*dx
    w3 = ri5*dy
    w4 = ri5*dz
  end if
  if (a == ii) then
    if (iorbs(ii) > 1) then
      c(2) = c(2) + qe(jj)*w5
      c(4) = c(4) + qe(jj)*w6
      c(7) = c(7) + qe(jj)*w7
    end if
    if (iorbs(jj) > 1) then
      c(1) = c(1) + ptot(j1 + 2)*w2*2
      c(1) = c(1) + ptot(j1 + 4)*w3*2
      c(1) = c(1) + ptot(j1 + 7)*w4*2
      if (iorbs(ii) > 1) then
        c(3) = c(3) + ptot(j1 + 2)*w2*2
        c(3) = c(3) + ptot(j1 + 4)*w3*2
        c(3) = c(3) + ptot(j1 + 7)*w4*2
        c(6) = c(6) + ptot(j1 + 2)*w2*2
        c(6) = c(6) + ptot(j1 + 4)*w3*2
        c(6) = c(6) + ptot(j1 + 7)*w4*2
        c(10) = c(10) + ptot(j1 + 2)*w2*2
        c(10) = c(10) + ptot(j1 + 4)*w3*2
        c(10) = c(10) + ptot(j1 + 7)*w4*2
      end if
    end if
  else
    if (iorbs(jj) > 1) then
      c(2) = c(2) + qe(ii)*w2
      c(4) = c(4) + qe(ii)*w3
      c(7) = c(7) + qe(ii)*w4
    end if
    if (iorbs(ii) > 1) then
      c(1) = c(1) + ptot(i1 + 2)*w5*2
      c(1) = c(1) + ptot(i1 + 4)*w6*2
      c(1) = c(1) + ptot(i1 + 7)*w7*2
      if (iorbs(jj) > 1) then
        c(3) = c(3) + ptot(i1 + 2)*w5*2
        c(3) = c(3) + ptot(i1 + 4)*w6*2
        c(3) = c(3) + ptot(i1 + 7)*w7*2
        c(6) = c(6) + ptot(i1 + 2)*w5*2
        c(6) = c(6) + ptot(i1 + 4)*w6*2
        c(6) = c(6) + ptot(i1 + 7)*w7*2
        c(10) = c(10) + ptot(i1 + 2)*w5*2
        c(10) = c(10) + ptot(i1 + 4)*w6*2
        c(10) = c(10) + ptot(i1 + 7)*w7*2
      end if
    end if
  end if
end subroutine acc_fz_far
#endif

subroutine acc_fock_compare(f, fg, tg)
! MOPAC_ACC_FOCK=test: report the largest difference between the CPU (f) and GPU (fg) Fock matrices.
  use molkst_C, only : mpack
  use chanel_C, only : iw
  implicit none
  double precision, intent(in) :: f(mpack), fg(mpack), tg
  integer :: i, imax
  double precision :: d, dmax, fmax
  dmax = 0.d0
  fmax = 0.d0
  imax = 1
  do i = 1, mpack
    d = abs(f(i) - fg(i))
    if (d > dmax) then
      dmax = d
      imax = i
    end if
    fmax = max(fmax, abs(f(i)))
  end do
  write (iw, '(a,es10.3,a,i0,a,es10.3,a,f8.3,a)') ' ACC FOCK TEST: max |F(CPU) - F(GPU)| ', dmax, ' eV at ', imax, &
    ' (max |F| ', fmax, ');  GPU part ', tg, ' s'
  flush (iw)
end subroutine acc_fock_compare
