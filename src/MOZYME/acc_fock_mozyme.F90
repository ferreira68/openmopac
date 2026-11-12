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
! GPU paths (Fock build here, diagg1 and density elsewhere) are on in an OpenACC build whenever a
! usable device is present (acc_gpu_mode): the keyword NOGPU turns them all off, SETGPU=n selects the
! device, and the environment variables MOPAC_ACC_FOCK, MOPAC_ACC_DIAGG and MOPAC_ACC_DENS set to 0
! turn one path off, or set to "test" also run the CPU code and write the largest difference to the
! output on every call. Without OpenACC, the modes are 0 and the GPU routines are empty.

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
! with suffix KB/MB/GB) caps the memory the plan may use (for a device shared with other jobs; a small
! limit also exercises the fallback levels). A failed acc_malloc later disables the GPU paths (CPU from
! then on) instead of stopping the program.
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
! One decision per geometry.
  use iso_c_binding, only : c_ptr, c_null_ptr, c_size_t, c_long_long, c_int, c_char, c_funptr
  implicit none
  integer, allocatable :: nij(:,:)
  integer :: key_numcal = -1, key_mpack = -1, key_numat = -1
  integer :: nij_gen = 0   ! incremented on every rebuild; consumers compare with the generation they used
  type(c_ptr) :: fd_c = c_null_ptr, pd_c = c_null_ptr, pacc_c = c_null_ptr, hd_c = c_null_ptr
  integer :: dev_len = 0, pacc_len = 0, hd_len = 0
! hd: device copy of the core Hamiltonian h for acc_helecz, valid while hd_ok (h does not change within
! an SCF; acc_h_invalidate is called at every entry to iter_for_MOZYME).
  logical :: hd_ok = .false.
! eimp_dev: the host eimp was skipped (acc_eimp_skip) because the GPU diagg1 applies it to the device
! copy of P; any host path that needs the eimp values first calls acc_eimp_host_fallback.
  logical :: eimp_dev = .false.
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
    integer(c_int) function c_acc_get_num_devices(devtype) bind(C, name="acc_get_num_devices")
      import :: c_int
      integer(c_int), value :: devtype
    end function c_acc_get_num_devices
    subroutine c_acc_set_device_num(devnum, devtype) bind(C, name="acc_set_device_num")
      import :: c_int
      integer(c_int), value :: devnum, devtype
    end subroutine c_acc_set_device_num
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
    integer(c_int) function cu_device_get_name(name, len, device) bind(C)  ! cuDeviceGetName(char*, int, CUdevice)
      import :: c_int, c_char
      character(kind=c_char), dimension(*) :: name
      integer(c_int), value :: len, device
    end function cu_device_get_name
    integer(c_int) function cu_host_register(p, bytes, flags) bind(C)  ! cuMemHostRegister(void*, size_t, unsigned)
      import :: c_int, c_ptr, c_size_t
      type(c_ptr), value :: p
      integer(c_size_t), value :: bytes
      integer(c_int), value :: flags
    end function cu_host_register
    integer(c_int) function cu_host_unregister(p) bind(C)              ! cuMemHostUnregister(void*)
      import :: c_int, c_ptr
      type(c_ptr), value :: p
    end function cu_host_unregister
  end interface
! Page-locked (pinned) host arrays. The host copies of F, P and H (common_arrays f, p, h; 611 MB each
! for whole PTP1B at CUTOFF=100) are copied to or from the device on every SCF iteration. From pageable
! memory the driver stages each copy through a bounce buffer (about 7 GB/s, RTX 5090 Laptop); pinned,
! the copy is a direct DMA. acc_to_device / acc_from_device pin a common array on its first transfer
! while pin_scope is set, that is, during one iter_for_MOZYME call (acc_pin_begin at its entry,
! acc_pin_end after every call). Within an SCF these arrays are never reallocated (they are only
! reallocated in setup_mopac_arrays, set_up_MOZYME_arrays, add_more_interactions (from hcore) and
! writmo); a pinned range must be unpinned before it is freed, because a new array at the same address
! would otherwise be read through the old, still-locked pages. Only common f, p and h are pinned.
  integer, parameter :: pin_max = 4
  logical :: pin_scope = .false.
  integer :: pin_n = 0, pin_state = 0        ! pin_state: 0 not initialized, 1 available, -1 unavailable
  type(c_ptr) :: pin_ptr(pin_max)
  integer(8) :: pin_lo(pin_max) = 0, pin_hi(pin_max) = 0
  logical :: pin_said(3) = .false.          ! the first pinning of f, p, h is reported
  procedure(cu_host_register), pointer :: pin_reg => null()
  procedure(cu_host_unregister), pointer :: pin_unreg => null()
! Execution-path report, written in every build and every job (lines " MOPAC execution path: ..."):
! at the start of the job (acc_path_start), after the device-memory plan (acc_mozyme_plan) and
! whenever the split between GPU and CPU changes (acc_path_update): full GPU, GPU/CPU hybrid (what runs
! on the CPU, and why) or CPU (why). The lines start with "MOPAC", which the MOPAC tests skip.
  character(len=400) :: path_last = ''
  character(len=200) :: gpu_why = ''        ! why no device is usable (acc_gpu_usable)
  character(len=160) :: gpu_desc = ''       ! the device used
  character(len=200) :: path_notes = ''     ! steps kept on the CPU for reasons other than memory (acc_path_note)
  character(len=120) :: path_mem = ''       ! device-memory reason of the plan, if it shed a GPU path
  logical :: path_planned = .false., path_want_f = .false., path_want_g = .false., path_want_d = .false.
  integer :: path_nsd = 0, path_ndd = 0
contains
  logical function acc_sp_atom(z, io)
! An atom (atomic number z, io orbitals) with an s or s,p basis and no d-orbital terms.
    use parameters_C, only : dorbs, iod
    integer, intent(in) :: z, io
    acc_sp_atom = (io == 1 .or. io == 4) .and. .not. dorbs(z) .and. iod(z) == 0
  end function acc_sp_atom

  subroutine acc_sp_order(numat, nat, iorbs, pos, ord, nsp)
! Positions of the s,p atoms in the tiles of the GPU Fock build: heavy (s,p) atoms first, then
! hydrogen-like (s) atoms, each in atom order, so that a tile pair holds pairs of one kind (heavy-heavy,
! heavy-light or light-light) and the threads of a warp follow the same branch with similar costs.
! pos(a) = 0 for an atom that is not s,p. ord(1:nsp) is the inverse.
    integer, intent(in) :: numat, nat(numat), iorbs(numat)
    integer, intent(out) :: pos(numat), ord(numat), nsp
    integer :: a, pass
    nsp = 0
    pos(1:numat) = 0
    do pass = 1, 2
      do a = 1, numat
        if (.not. acc_sp_atom(nat(a), iorbs(a))) cycle
        if ((pass == 1) .neqv. (iorbs(a) == 4)) cycle
        nsp = nsp + 1
        pos(a) = nsp
        ord(nsp) = a
      end do
    end do
  end subroutine acc_sp_order

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
      write (iw, '(a)') ' MOPAC GPU: device memory exhausted allocating the density accumulator; density on the CPU'
      acc_mozyme_pacc = .false.
      return
    end if
    pacc_len = n
#endif
  end function acc_mozyme_pacc

  logical function acc_mozyme_hbuf(n)
! Makes hd hold at least n doubles. On failure, helecz stays on the CPU (.false.).
    use iso_c_binding, only : c_associated
    use chanel_C, only : iw
    integer, intent(in) :: n
    acc_mozyme_hbuf = .false.
#ifdef _OPENACC
    acc_mozyme_hbuf = .true.
    if (n <= hd_len) return
    if (c_associated(hd_c)) call c_acc_free(hd_c)
    hd_len = 0
    hd_ok = .false.
    hd_c = c_acc_malloc(int(n, c_long_long)*8_c_long_long)
    if (.not. c_associated(hd_c)) then
      hd_c = c_null_ptr
      write (iw, '(a)') ' MOPAC GPU: device memory exhausted allocating the copy of H; HELECZ on the CPU'
      acc_mozyme_hbuf = .false.
      return
    end if
    hd_len = n
#endif
  end function acc_mozyme_hbuf

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
    write (iw, '(3a)') ' MOPAC GPU: device memory exhausted ', what, '; MOZYME continues on the CPU'
    call acc_path_update('device memory exhausted '//what)
#endif
  end subroutine acc_mozyme_give_up

  subroutine acc_mozyme_release()
! Frees the device buffers owned by this module (fd, pd, pacc, hd).
    use iso_c_binding, only : c_associated
#ifdef _OPENACC
    if (c_associated(fd_c)) call c_acc_free(fd_c)
    if (c_associated(pd_c)) call c_acc_free(pd_c)
    if (c_associated(pacc_c)) call c_acc_free(pacc_c)
    if (c_associated(hd_c)) call c_acc_free(hd_c)
    fd_c = c_null_ptr
    pd_c = c_null_ptr
    pacc_c = c_null_ptr
    hd_c = c_null_ptr
    dev_len = 0
    pacc_len = 0
    hd_len = 0
    hd_ok = .false.
    fd_ok = .false.
    pd_ok = .false.
#endif
  end subroutine acc_mozyme_release

  logical function acc_report()
! Routine reports of the GPU code (device, memory plan, decisions, pinning) are written only with the
! keyword TIMES or DEBUG, so that normal output matches the CPU program line for line (the MOPAC tests
! compare outputs line by line). Warnings and the "test" comparisons are always written.
    use molkst_C, only : keywrd
    acc_report = index(keywrd, " TIMES") /= 0 .or. index(keywrd, " DEBUG") /= 0
  end function acc_report

  integer function acc_gpu_mode(envname)
! Mode of one GPU path for this job: 0 CPU, 1 GPU, 2 both, compared ("test"). Off with the keyword
! NOGPU or without a usable device (acc_gpu_usable); otherwise on, unless the path's environment
! variable (MOPAC_ACC_FOCK, MOPAC_ACC_DIAGG, MOPAC_ACC_DENS) is "0", or "test".
    use molkst_C, only : keywrd
    character(len=*), intent(in) :: envname
    character(len=16) :: env
    integer :: stat
    acc_gpu_mode = 0
#ifdef _OPENACC
    if (index(keywrd, " NOGPU") /= 0) return
    if (.not. acc_gpu_usable()) return
    acc_gpu_mode = 1
    call get_environment_variable(envname, env, status=stat)
    if (stat == 0) then
      if (trim(env) == '0') acc_gpu_mode = 0
      if (trim(env) == 'test') acc_gpu_mode = 2
    end if
#endif
  end function acc_gpu_mode

  logical function acc_gpu_usable()
! Once per process: selects the device (keyword SETGPU=n, n = 1, 2, ... among the visible devices;
! default the first) and checks that this build has device code it can run: one of the compute
! capabilities in MOPAC_ACC_CCS (CMake GPU_OPENACC_ARCH) with the same major version and a minor
! version no higher than the device's. Otherwise MOZYME runs on the CPU. The outcome is written once.
! The device cannot change later in the same process (the device buffers live there).
    use molkst_C, only : keywrd
    use chanel_C, only : iw
    integer, save :: state = 0                 ! 0 not yet checked, 1 usable, -1 not usable
#ifdef _OPENACC
#ifndef MOPAC_ACC_CCS
#define MOPAC_ACC_CCS 86,120
#endif
    integer, parameter :: built(*) = [MOPAC_ACC_CCS]
    integer :: ndev, i, n, dev, cc, k
    double precision, external :: reada
    logical :: ok
    character(len=64) :: bad
    bad = ''
    if (state == 0) then
      state = -1
      ndev = c_acc_get_num_devices(4_c_int)            ! acc_device_nvidia
      if (ndev < 1) then
        gpu_why = 'no GPU visible'
      else
        i = index(keywrd, " SETGPU=")
        if (i /= 0) then
          n = nint(reada(keywrd, i))
          if (n >= 1 .and. n <= ndev) then
            call c_acc_set_device_num(int(n - 1, c_int), 4_c_int)
          else
            write (bad, '(a,i0,a,i0,a)') '; SETGPU=', n, ' is not one of the ', ndev, ' visible devices'
          end if
        end if
        dev = c_acc_get_device_num(4_c_int)
        cc = acc_device_cc(dev)
        ok = .false.
        do k = 1, size(built)
          if (cc > 0 .and. cc/10 == built(k)/10 .and. mod(cc, 10) >= mod(built(k), 10)) ok = .true.
        end do
        write (gpu_desc, '(a,i0,3a,i0,a,i0,a)') 'device ', dev + 1, ', ', trim(acc_device_name(dev)), &
          ', compute capability ', cc/10, '.', mod(cc, 10), trim(bad)
        if (ok) then
          state = 1
        else
          write (gpu_why, '(3a,20(i0,a))') 'GPU ', trim(gpu_desc), ' has no code in this build; built for compute capability ', &
            (built(k)/10, '.', mod(built(k), 10), ' ', k = 1, size(built))
        end if
      end if
    end if
    acc_gpu_usable = (state == 1)
#else
    acc_gpu_usable = .false.
#endif
  end function acc_gpu_usable

  character(len=96) function acc_device_name(devnum)
! Name of CUDA device devnum (driver API through dlopen), or "unknown GPU".
    use iso_c_binding, only : c_associated, c_f_procpointer, c_null_char
    integer, intent(in) :: devnum
    type(c_ptr) :: h
    type(c_funptr) :: fp1, fp2
    procedure(cu_device_get), pointer :: get_dev
    procedure(cu_device_get_name), pointer :: get_name
    integer(c_int) :: dev
    character(kind=c_char) :: buf(96)
    integer :: i
    acc_device_name = 'unknown GPU'
#ifdef _OPENACC
    h = c_dlopen('libcuda.so.1'//c_null_char, 2_c_int)
    if (.not. c_associated(h)) return
    fp1 = c_dlsym(h, 'cuDeviceGet'//c_null_char)
    fp2 = c_dlsym(h, 'cuDeviceGetName'//c_null_char)
    if (.not. c_associated(fp1) .or. .not. c_associated(fp2)) return
    call c_f_procpointer(fp1, get_dev)
    call c_f_procpointer(fp2, get_name)
    if (get_dev(dev, int(devnum, c_int)) /= 0) return
    buf = c_null_char
    if (get_name(buf, 95_c_int, dev) /= 0) return
    acc_device_name = ''
    do i = 1, 95
      if (buf(i) == c_null_char) exit
      acc_device_name(i:i) = buf(i)
    end do
#endif
  end function acc_device_name

  subroutine acc_path_say(text)
! Writes the execution path, unless it is the one written last in this job.
    use chanel_C, only : iw
    character(len=*), intent(in) :: text
    if (trim(text) == trim(path_last)) return
    path_last = text
    write (iw, '(2a)') ' MOPAC execution path: ', trim(text)
  end subroutine acc_path_say

  subroutine acc_path_start()
! Called for every job after the keywords are written (readmo): the execution path at the start
! and, if MOZYME will not use the GPU, why. For MOZYME on the GPU the split between GPU and CPU is
! reported after the device-memory plan of the first SCF (acc_path_update).
    use molkst_C, only : keywrd, mozyme
    path_last = ''
    path_notes = ''
    path_mem = ''
    path_planned = .false.
#ifdef _OPENACC
    if (.not. mozyme) then
#ifdef MOPAC_MAGMA
      call acc_path_say('conventional SCF on the CPU; dense eigensolves on the GPU (MAGMA) where they fit')
#else
      call acc_path_say('CPU (conventional SCF; this build runs MOZYME on the GPU)')
#endif
    else if (index(keywrd, " NOGPU") /= 0) then
      call acc_path_say('CPU (keyword NOGPU)')
    else if (.not. acc_gpu_usable()) then
      call acc_path_say('CPU ('//trim(gpu_why)//')')
    else if (acc_gpu_mode("MOPAC_ACC_FOCK") == 0 .and. acc_gpu_mode("MOPAC_ACC_DIAGG") == 0 .and. &
      acc_gpu_mode("MOPAC_ACC_DENS") == 0) then
      call acc_path_say('CPU (MOPAC_ACC_FOCK, MOPAC_ACC_DIAGG and MOPAC_ACC_DENS are set to 0)')
    else
      call acc_path_say('MOZYME on the GPU ('//trim(gpu_desc)// &
        '); the split between GPU and CPU follows from the device memory, at the first SCF')
    end if
#else
    if (.false.) write (path_last, '(2a)') trim(keywrd), mozyme   ! no GPU code in this build
#ifdef MOPAC_MAGMA
    call acc_path_say('CPU, except dense eigensolves on the GPU (MAGMA) where they fit')
#else
    call acc_path_say('CPU (built without GPU support)')
#endif
#endif
  end subroutine acc_path_start

  subroutine acc_path_note(what)
! A GPU path is not used for a step for a reason other than device memory (e.g. partial Fock updates):
! add it to the CPU list of the execution path (once per job).
    character(len=*), intent(in) :: what
    if (index(path_notes, what) /= 0) return
    if (len_trim(path_notes) > 0) then
      path_notes = trim(path_notes)//'; '//what
    else
      path_notes = what
    end if
    if (path_planned) call acc_path_update('')
  end subroutine acc_path_note

  subroutine acc_path_update(why)
! Reports the current split between GPU and CPU from the plan's flags (allow_*), the paths turned
! off by MOPAC_ACC_*=0, the notes, and why (the reason for a change), if it differs from the last line.
    character(len=*), intent(in) :: why
    character(len=400) :: cpu, text
    logical :: any_gpu
    cpu = ''
    if (.not. path_want_f) call add('Fock build (MOPAC_ACC_FOCK=0)')
    if (.not. path_want_d) call add('density (MOPAC_ACC_DENS=0)')
    if (.not. path_want_g) call add('DIAGG1 (MOPAC_ACC_DIAGG=0)')
    if (path_want_f .and. .not. allow_fock) then
      call add('Fock build')
    else if (allow_fock .and. path_nsd > 0 .and. .not. allow_sdgpu) then
      call add('NDDO pairs with d orbitals in the Fock build')
    else if (allow_fock .and. path_ndd > 0 .and. .not. allow_ddgpu) then
      if (dd_pending) then
        call add('d,d NDDO pairs in the Fock build (moved to the GPU after the first Fock builds if that pays)')
      else
        call add('d,d NDDO pairs in the Fock build')
      end if
    end if
    if (path_want_d .and. .not. allow_dens) call add('density')
    if (path_want_g .and. .not. allow_diagg) call add('DIAGG1')
    if (len_trim(path_notes) > 0) call add(trim(path_notes))
    any_gpu = allow_fock .or. allow_dens .or. allow_diagg
    if (.not. any_gpu) then
      text = 'CPU'
      if (len_trim(path_mem) > 0) text = 'CPU ('//trim(path_mem)//')'
    else if (len_trim(cpu) == 0) then
      text = 'full GPU (every MOZYME step with GPU code: Fock build, density, DIAGG1, energy)'
    else
      text = 'GPU/CPU hybrid; on the CPU: '//trim(cpu)
      if (len_trim(path_mem) > 0) text = trim(text)//' ('//trim(path_mem)//')'
    end if
    if (len_trim(why) > 0) text = trim(text)//' - '//why
    call acc_path_say(text)
  contains
    subroutine add(item)
      character(len=*), intent(in) :: item
      if (len_trim(cpu) > 0) then
        cpu = trim(cpu)//', '//item
      else
        cpu = item
      end if
    end subroutine add
  end subroutine acc_path_update

  integer function acc_device_cc(devnum)
! Compute capability of CUDA device devnum as 10*major + minor, from the driver API (libcuda through
! dlopen), or 0 if unavailable.
    use iso_c_binding, only : c_associated, c_f_procpointer, c_null_char
    integer, intent(in) :: devnum
    type(c_ptr) :: h
    type(c_funptr) :: fp1, fp2
    procedure(cu_device_get), pointer :: get_dev
    procedure(cu_device_get_attribute), pointer :: get_attr
    integer(c_int) :: dev, major, minor
    acc_device_cc = 0
#ifdef _OPENACC
    h = c_dlopen('libcuda.so.1'//c_null_char, 2_c_int)        ! RTLD_NOW
    if (.not. c_associated(h)) return
    fp1 = c_dlsym(h, 'cuDeviceGet'//c_null_char)
    fp2 = c_dlsym(h, 'cuDeviceGetAttribute'//c_null_char)
    if (.not. c_associated(fp1) .or. .not. c_associated(fp2)) return
    call c_f_procpointer(fp1, get_dev)
    call c_f_procpointer(fp2, get_attr)
    if (get_dev(dev, int(devnum, c_int)) /= 0) return
    if (get_attr(major, 75_c_int, dev) /= 0) return             ! CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    if (get_attr(minor, 76_c_int, dev) /= 0) return             ! CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    acc_device_cc = 10*major + minor
#endif
  end function acc_device_cc

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
    integer(8) :: npair, npair_d, npair_sd, npair_dd, lcocc, lcvir, licocc, licvir, ntpair
    logical, allocatable :: tmark(:,:)
    integer :: nt, nsp
    integer, allocatable :: spos(:), sord(:)
    integer :: i, j, dev, k
    double precision, parameter :: gb = 1024.d0**3
    character(len=64) :: enabled
    integer :: k0
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
    nt = (numat + 31)/32
    allocate (tmark(nt, nt), spos(numat), sord(numat))
    tmark = .false.
    call acc_sp_order(numat, nat, iorbs, spos, sord, nsp)
    do i = 2, numat
      if (iorbs(i) == 0) cycle
      do j = 1, i - 1
        if (iorbs(j) == 0 .or. nijbo(i, j) < 0) cycle
        npair = npair + 1
        select case (acc_pair_class(nat(i), iorbs(i), nat(j), iorbs(j)))
        case (0)                                             ! tile pairs of acc_fz2n (tile_n = 32)
          tmark((max(spos(i), spos(j)) - 1)/32 + 1, (min(spos(i), spos(j)) - 1)/32 + 1) = .true.
        case (1)
          npair_sd = npair_sd + 1
        case (2)
          npair_dd = npair_dd + 1
        end select
      end do
    end do
    npair_d = npair_sd + npair_dd
    ntpair = count(tmark)
    deallocate (tmark, spos, sord)
    lcocc = 0; lcvir = 0; licocc = 0; licvir = 0
    if (allocated(cocc)) lcocc = size(cocc, kind=8)
    if (allocated(cvir)) lcvir = size(cvir, kind=8)
    if (allocated(icocc)) licocc = size(icocc, kind=8)
    if (allocated(icvir)) licvir = size(icvir, kind=8)
    b_nij = 4_8*int(numat, 8)**2
    b_fp = 24_8*int(mpack, 8)              ! F, P and H
    b_pacc = 8_8*int(mpack, 8)
    b_lists = 5120_8*ntpair + 8_8*ntpair + 760_8*npair_d + 24_8*numat   ! tile partials (2 x 32 x 10 doubles) + lists
    b_cand = 30_8*max(1000000_8, 2_8*int(fmo_dim, 8))     ! candidate buffer (20 bytes each) plus growth
    b_lmo_g = 16_8*(lcocc + lcvir) + 24_8*(licocc + licvir)
    b_lmo_d = 8_8*lcocc + 8_8*licocc
    lmem_sp = stack_rotate_sp*slots
    lmem_sd = stack_rotate_spd*slots
    lmem_dd = stack_rotate*slots
! Levels (module header): 1 everything; 2 d,d pairs on the CPU; 3 all d pairs on the CPU; 4 no
! density; 5 Fock only; 6 nothing. Level 1 is chosen here only if the d,d reservation is already
! held; otherwise it is decided after the first Fock builds (acc_dd_decide).
    k0 = 2
    if (npair_dd > 0 .and. lmem_held >= lmem_dd) k0 = 1
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
    path_want_f = want_fock
    path_want_g = want_diagg
    path_want_d = want_dens
    path_nsd = int(npair_sd)
    path_ndd = int(npair_dd)
    path_mem = ''
    if (k > max(k0, 2) .or. (k == 2 .and. k0 == 1)) then
      f = want_fock; g = want_diagg; d = want_dens; sdg = f .and. npair_sd > 0; ddg = f .and. npair_dd > 0
      write (path_mem, '(a,f0.2,a,f0.2,a)') 'device memory: ', dble(need())/gb, ' GB needed for all GPU paths, ', &
        dble(max(0_8, avail - headroom))/gb, ' GB available'
    end if
    dd_pending = k == 2 .and. k0 == 2 .and. npair_dd > 0 .and. allow_sdgpu
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
    path_planned = .true.
    call acc_path_update('')
    if (acc_report()) then
      write (iw, '(a,f7.3,a,f7.3,a)', advance='no') ' MOPAC GPU: device memory free ', dble(free)/gb, ' of ', &
        dble(total)/gb, ' GB'
      if (limit >= 0) write (iw, '(a,f7.3,a)', advance='no') ' (MOPAC_ACC_MEM_LIMIT ', dble(limit)/gb, ' GB)'
      write (iw, '(a,f7.3,a,f7.3,a,f7.3,a,f7.3,a,f7.3,a,f7.3,a,f7.3,a,i0,a)') '; needed (GB): Fock ', &
        dble(b_nij + b_fp + b_lists + lmem_sp)/gb, ', +s,p-d pairs ', dble(lmem_sd - lmem_sp)/gb, ', +d,d pairs ', &
        dble(lmem_dd - lmem_sd)/gb, ', +DIAGG1 ', dble(b_cand + b_lmo_g)/gb, ', +density ', dble(b_pacc + b_lmo_d)/gb, &
        ', headroom ', dble(headroom)/gb, ', available ', dble(avail)/gb, ' (', slots, ' thread slots)'
      write (iw, '(a,i0,a,i0,a,i0)', advance='no') ' MOPAC GPU: NDDO pairs ', npair, ', of which s,p-d ', npair_sd, &
        ', d,d ', npair_dd
      if (dd_pending) write (iw, '(a)', advance='no') &
        ' (d,d pairs on the CPU; moved to the GPU after the first Fock builds if that pays)'
      write (iw, '(a)') ''
      write (iw, '(a,i0,2a)') ' MOPAC GPU: level ', k, '; GPU paths enabled:', trim(enabled)
      flush (iw)
    end if
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
    logical :: rep
    character(len=200) :: msg
    if (t_dd_cpu < 0.d0 .or. t_iter <= 0.d0) return        ! not measured yet
    if (nsd > 0 .and. t_sd_gpu < 0.d0) return
    dd_pending = .false.
    est = 0.d0
    if (nsd > 0) est = dble(ndd)*(t_sd_gpu/dble(nsd))*k_dd
    saving = t_dd_cpu - est
    need = max(0_8, lmem_dd - lmem_held)
    free = acc_free_memory()
    rep = acc_report()
    if (rep) write (iw, '(a,f8.4,a,f8.4,a,f8.3,a)', advance='no') ' MOPAC GPU: d,d pairs: CPU ', t_dd_cpu, &
      ' s per Fock build, GPU (estimated) ', est, ' s, SCF iteration ', t_iter, ' s'
    if (saving <= f_dd*t_iter) then
      if (rep) write (iw, '(a)') '; stay on the CPU (saving under 5 % of an iteration)'
      write (msg, '(a,f0.4,a,f0.4,a)') 'kept there: ', t_dd_cpu, &
        ' s per Fock build on the CPU, under 5 % of an SCF iteration (', t_iter, ' s)'
    else if (free - need < plan_headroom) then
      if (rep) write (iw, '(a,f6.2,a,f6.2,a)') '; would pay, but the kernel reservation (', dble(need)/1024.d0**3, &
        ' GB) does not fit (free ', dble(free)/1024.d0**3, ' GB): stay on the CPU'
      write (msg, '(a,f0.2,a,f0.2,a)') 'kept there: their GPU kernel needs ', dble(need)/1024.d0**3, &
        ' GB of device memory, ', dble(free)/1024.d0**3, ' GB free'
    else
      allow_ddgpu = .true.
      if (rep) write (iw, '(a,f6.2,a)') '; moved to the GPU (reservation ', dble(need)/1024.d0**3, ' GB)'
      write (msg, '(a,f0.4,a,f0.4,a)') 'from the next Fock build; d,d NDDO pairs moved to the GPU (CPU ', t_dd_cpu, &
        ' s per Fock build, GPU estimated ', est, ' s)'
    end if
    if (rep) flush (iw)
    call acc_path_update(trim(msg))
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
    if (acc_report()) write (iw, '(3a,f6.2,a,f6.2,a)') ' MOPAC GPU: ', what, ' kernel stack reservation: predicted ', &
      dble(expect)/1024.d0**3, ' GB, measured ', dble(drop)/1024.d0**3, ' GB'
    if (acc_report() .and. abs(dble(drop) - dble(expect)) > 0.1d0*dble(expect)) write (iw, '(3a)') &
      ' MOPAC GPU: WARNING: the stack reservation differs from the prediction by more than 10 %; update the ', &
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
    if (pin_scope) call acc_pin(c_loc(a), int(n, 8)*8_8)
    call c_acc_memcpy_to_device(dev, c_loc(a), int(n, c_long_long)*8_c_long_long)
#endif
  end subroutine acc_to_device

  subroutine acc_from_device(a, dev, n)
    use iso_c_binding, only : c_loc
    type(c_ptr), intent(in) :: dev
    integer, intent(in) :: n
    double precision, intent(inout), target :: a(n)
#ifdef _OPENACC
    if (pin_scope) call acc_pin(c_loc(a), int(n, 8)*8_8)
    call c_acc_memcpy_from_device(c_loc(a), dev, int(n, c_long_long)*8_c_long_long)
#endif
  end subroutine acc_from_device

  subroutine acc_pin(p, bytes)
! Pins the whole common array (f, p or h) that holds the host range [p, p+bytes), unless it is
! pinned already, the range is in none of them, or it is smaller than 16 MB (see pin_scope).
    use iso_c_binding, only : c_intptr_t, c_associated, c_f_procpointer, c_null_char, c_loc
    use common_arrays_C, only : fcom => f, pcom => p, hcom => h
    use chanel_C, only : iw
    type(c_ptr), intent(in) :: p
    integer(8), intent(in) :: bytes
#ifdef _OPENACC
    integer(8) :: lo, hi, alo, ahi, tc0, tc1, trate
    integer :: k, which, stat
    type(c_ptr) :: h, base
    type(c_funptr) :: fp1, fp2
    if (bytes < 16_8*1024_8**2) return
    lo = transfer(p, 0_c_intptr_t)
    hi = lo + bytes
    do k = 1, pin_n
      if (lo >= pin_lo(k) .and. hi <= pin_hi(k)) return
    end do
    if (pin_state == 0) then
      pin_state = -1
      h = c_dlopen('libcuda.so.1'//c_null_char, 2_c_int)        ! RTLD_NOW
      if (.not. c_associated(h)) return
      fp1 = c_dlsym(h, 'cuMemHostRegister_v2'//c_null_char)
      fp2 = c_dlsym(h, 'cuMemHostUnregister'//c_null_char)
      if (.not. c_associated(fp1) .or. .not. c_associated(fp2)) return
      call c_f_procpointer(fp1, pin_reg)
      call c_f_procpointer(fp2, pin_unreg)
      pin_state = 1
    end if
    if (pin_state /= 1 .or. pin_n >= pin_max) return
    which = 0
    alo = 0
    ahi = 0
    do k = 1, 3
      if (k == 1 .and. allocated(fcom)) then
        alo = loc(fcom(1))               ! loc: the common arrays have no TARGET attribute
        base = transfer(int(alo, c_intptr_t), base)
        ahi = alo + 8_8*size(fcom, kind=8)
      else if (k == 2 .and. allocated(pcom)) then
        alo = loc(pcom(1))               ! loc: the common arrays have no TARGET attribute
        base = transfer(int(alo, c_intptr_t), base)
        ahi = alo + 8_8*size(pcom, kind=8)
      else if (k == 3 .and. allocated(hcom)) then
        alo = loc(hcom(1))               ! loc: the common arrays have no TARGET attribute
        base = transfer(int(alo, c_intptr_t), base)
        ahi = alo + 8_8*size(hcom, kind=8)
      else
        cycle
      end if
      if (lo >= alo .and. hi <= ahi) then
        which = k
        exit
      end if
    end do
    if (which == 0) return
    do k = 1, pin_n                     ! cannot happen while the arrays are not freed within the scope
      if (alo < pin_hi(k) .and. ahi > pin_lo(k)) then
        write (iw, '(a)') ' MOPAC GPU: pinned host ranges overlap; pinning stopped'
        pin_state = -1
        return
      end if
    end do
    call system_clock(tc0, trate)
    stat = pin_reg(base, int(ahi - alo, c_size_t), 1_c_int)    ! CU_MEMHOSTREGISTER_PORTABLE
    call system_clock(tc1)
    if (stat /= 0) then
      write (iw, '(a,i0,a)') ' MOPAC GPU: cuMemHostRegister failed (CUresult ', stat, '); transfers stay pageable'
      pin_state = -1
      return
    end if
    pin_n = pin_n + 1
    pin_ptr(pin_n) = base
    pin_lo(pin_n) = alo
    pin_hi(pin_n) = ahi
    if (.not. pin_said(which) .and. acc_report()) then
      write (iw, '(a,a,f8.3,a,f7.3,a)') ' MOPAC GPU: page-locked host array ', merge('f', merge('p', 'h', which == 2), &
        which == 1), dble(ahi - alo)/1024.d0**3, ' GB for device transfers (', dble(tc1 - tc0)/dble(trate), ' s)'
      pin_said(which) = .true.
    end if
#endif
  end subroutine acc_pin

  subroutine acc_pin_release()
! Unpins every pinned host range (end of the pin scope).
    integer :: k, stat
#ifdef _OPENACC
    do k = 1, pin_n
      stat = pin_unreg(pin_ptr(k))
    end do
#endif
    pin_n = 0
    pin_lo = 0
    pin_hi = 0
    pin_scope = .false.
  end subroutine acc_pin_release

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
! Mode of the GPU Fock build (acc_gpu_mode): 0 CPU, 1 GPU, 2 both, compared (MOPAC_ACC_FOCK=test).
  use acc_mozyme_C, only : acc_gpu_mode
  implicit none
  acc_fock_mode = acc_gpu_mode("MOPAC_ACC_FOCK")
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
    acc_path_note, acc_report, &
    fd_c, pd_c, fd_ok, pd_ok, acc_matches_device, allow_fock, held_lists, allow_sdgpu, allow_ddgpu, n_sdpairs, &
    n_ddpairs, lmem_sp, lmem_sd, lmem_dd, lmem_held, acc_pair_class, acc_lmem_begin, acc_lmem_end, &
    dd_pending, t_iter, t_last_build, t_sd_gpu, acc_dd_decide, acc_sp_order, &
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
  integer, save :: my_gen = -1, ntile = 0, ntp = 0
! s,p NDDO pairs in tiles of tile_n consecutive positions of the s,p atoms (acc_sp_order: heavy atoms
! first, so that tile pairs are of one kind): tile pairs (ta >= tb) that hold at least one such pair,
! per tile the list of its tile pairs (tlist from tstart: +tp, the tile is ta(tp); -tp, it is tb(tp);
! in increasing tp, + before -), the position of each atom (spos, 0 if not s,p) and the atom at each
! position (sord), and the partial sums (part).
  integer, parameter :: tile_n = 32
  integer, allocatable, save :: ta(:), tb(:), tstart(:), tlist(:), spos(:), sord(:)
  integer, save :: nsp = 0
  double precision, allocatable, save :: part(:,:,:,:)
  logical, allocatable :: mark(:,:)
  integer :: x, y, nent
! The NDDO pairs with d orbitals (dpi, dpj, ... ) are module arrays of acc_mozyme_C (dp_*).
  logical, save :: unsupported = .false.
  integer :: kd, m
  integer, allocatable :: dtmp(:)
  logical :: spi, spj, rebuilt
  integer :: a, b, ii, jj, k
  supported = .false.
  rebuilt = acc_mozyme_sync()   ! separate statement: Fortran need not evaluate a function inside .or.
  if ((rebuilt .or. my_gen /= nij_gen .or. .not. allow_fock) .and. allocated(ta)) then
    !$acc exit data delete(ta, tb, tstart, tlist, spos, sord, part, dpi, dpj, dcls, isd, idd, dstart, dlist, cdi, cdj)
    deallocate (ta, tb, tstart, tlist, spos, sord, part, dpi, dpj, dcls, isd, idd, dstart, dlist, cdi, cdj)
    held_lists = 0
    my_gen = -1
  end if
  if (.not. allow_fock) return            ! not enabled, or not enough device memory: CPU
  if (my_gen /= nij_gen) then
! New geometry or new set of interactions: rebuild the tile lists of the s,p NDDO pairs.
    allocate (spos(numat), sord(numat))
    call acc_sp_order(numat, nat, iorbs, spos, sord, nsp)
    ntile = max(1, (nsp + tile_n - 1)/tile_n)
    allocate (mark(ntile, ntile))
    mark = .false.
    do b = 1, numat                       ! nij(a, b) is contiguous in a
      if (spos(b) == 0) cycle
      y = (spos(b) - 1)/tile_n + 1
      do a = b + 1, numat
        if (spos(a) == 0) cycle
        if (nij(a, b) < 0) cycle
        x = (spos(a) - 1)/tile_n + 1
        mark(max(x, y), min(x, y)) = .true.
      end do
    end do
    ntp = 0
    do x = 1, ntile
      do y = 1, x
        if (mark(x, y)) ntp = ntp + 1
      end do
    end do
    allocate (ta(max(1, ntp)), tb(max(1, ntp)), tstart(ntile + 1), part(10, tile_n, 2, max(1, ntp)))
    ntp = 0
    tstart = 0
    nent = 0
    do x = 1, ntile
      do y = 1, x
        if (.not. mark(x, y)) cycle
        ntp = ntp + 1
        ta(ntp) = x
        tb(ntp) = y
        tstart(x) = tstart(x) + 1
        tstart(y) = tstart(y) + 1
        nent = nent + 2
      end do
    end do
    deallocate (mark)
    allocate (tlist(max(1, nent)), dtmp(ntile))
    m = 1
    do x = 1, ntile
      k = tstart(x)
      tstart(x) = m
      m = m + k
    end do
    tstart(ntile + 1) = m
    dtmp(1:ntile) = tstart(1:ntile)
    do k = 1, ntp
      tlist(dtmp(ta(k))) = k
      dtmp(ta(k)) = dtmp(ta(k)) + 1
      tlist(dtmp(tb(k))) = -k
      dtmp(tb(k)) = dtmp(tb(k)) + 1
    end do
    deallocate (dtmp)
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
    if (unsupported) call acc_path_note('Fock build (NDDO pairs of a kind the GPU code does not handle)')
    !$acc enter data copyin(ta, tb, tstart, tlist, spos, sord, dpi, dpj, dcls, isd, idd, dstart, dlist) create(part, cdi, cdj)
    n_sdpairs = nsd
    n_ddpairs = ndd
    held_lists = 4_8*(size(ta, kind=8) + size(tb, kind=8) + size(tstart, kind=8) + size(tlist, kind=8) + &
      size(spos, kind=8) + size(sord, kind=8) + size(dpi, kind=8) + size(dpj, kind=8) + &
      size(dcls, kind=8) + size(isd, kind=8) + size(idd, kind=8) + size(dstart, kind=8) + size(dlist, kind=8)) + &
      8_8*(size(part, kind=8) + size(cdi, kind=8) + size(cdj, kind=8))
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
    if (.not. p_reuse) write (iw, '(a)') ' MOPAC GPU Fock: device copy of P was stale; copied again'
  end if
  if (p_reuse) then
    nreuse = nreuse + 1
    if (nreuse == 1 .and. acc_report()) write (iw, '(a)') ' MOPAC GPU Fock: using the device copy of P from the GPU density build'
  else
    call acc_to_device(pd_c, ptot, mpack)
  end if
  call c_f_pointer(fd_c, fdev, [mpack])
  call c_f_pointer(pd_c, pdev, [mpack])
  if (ntp > 0) call acc_fz2n_tiles(numat, mpack, ntp, tile_n, nsp, fdev, pdev, iorbs, nat, coord, nij, sord, ta, tb, &
    part)
  call acc_fz2n_kernel(numat, mpack, fdev, pdev, qe, iorbs, nat, coord, nij, ntile, ntp, tile_n, spos, tstart, tlist, part)
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

subroutine acc_fz2n_tiles(numat, mpack, ntp, tile_n, nsp, f, ptot, iorbs, nat, coord, nij, sord, ta, tb, part)
! s,p NDDO pairs in blocks of s,p atom positions (acc_sp_order: heavy atoms first, so a tile pair holds
! pairs of one kind): one gang of tile_n threads per tile pair (ta, tb), ta >= tb; thread t owns the
! atom at position pa = (ta-1)*tile_n + t. In step s it takes the pair with the atom at position
! pb = (tb-1)*tile_n + 1 + mod(t-1+s, tile_n), so within a step the threads of the gang take different
! atoms b (in a diagonal tile, only pb < pa): each pair is computed once
! (acc_fz_pair_sp), its contribution to a is added to the thread's own sum (ca) and that to b to the
! gang's shared sum for b (sb) without atomics, in a fixed order. The two sums per atom are stored as
! the tile pair's partial sums (part(:,:,1,tp) for the atoms of ta, part(:,:,2,tp) for those of tb),
! which acc_fz2n_kernel adds per atom in a fixed order: the result is reproducible.
  implicit none
  integer, intent(in) :: numat, mpack, ntp, tile_n, nsp
  double precision, intent(inout) :: f(mpack)
  double precision, intent(in) :: ptot(mpack), coord(3, numat)
  integer, intent(in) :: iorbs(numat), nat(numat), nij(numat, numat), sord(numat), ta(ntp), tb(ntp)
  double precision, intent(out) :: part(10, tile_n, 2, ntp)
  integer :: tp, t, s, a, b, a0, b0, x, pa, pb
  double precision :: ca(10, 32), sb(10, 32), ci(10), cj(10)
  !$acc routine(acc_fz_pair_sp) seq
  !$acc data copyin(coord, iorbs, nat) present(nij, sord, ta, tb, part) deviceptr(f, ptot)
  !$acc parallel loop gang vector_length(32) private(ca, sb, a0, b0)
  do tp = 1, ntp
    a0 = (ta(tp) - 1)*tile_n
    b0 = (tb(tp) - 1)*tile_n
    !$acc loop vector private(x)
    do t = 1, tile_n
      do x = 1, 10
        ca(x, t) = 0.d0
        sb(x, t) = 0.d0
      end do
    end do
    !$acc loop seq
    do s = 0, tile_n - 1
      !$acc loop vector private(a, b, ci, cj, x, pa, pb)
      do t = 1, tile_n
        pa = a0 + t
        pb = b0 + 1 + mod(t - 1 + s, tile_n)
        if (pa <= nsp .and. pb < pa) then
          a = sord(pa)
          b = sord(pb)
          if (nij(max(a, b), min(a, b)) >= 0) then
! ii = max(a, b) for the integrals; ci goes to ii, cj to jj
            call acc_fz_pair_sp(max(a, b), min(a, b), numat, mpack, f, ptot, iorbs, nat, coord, nij, ci, cj)
            if (a > b) then
              do x = 1, 10
                ca(x, t) = ca(x, t) + ci(x)
                sb(x, pb - b0) = sb(x, pb - b0) + cj(x)
              end do
            else
              do x = 1, 10
                ca(x, t) = ca(x, t) + cj(x)
                sb(x, pb - b0) = sb(x, pb - b0) + ci(x)
              end do
            end if
          end if
        end if
      end do
    end do
    !$acc loop vector private(x)
    do t = 1, tile_n
      do x = 1, 10
        part(x, t, 1, tp) = ca(x, t)
        part(x, t, 2, tp) = sb(x, t)
      end do
    end do
  end do
  !$acc end data
end subroutine acc_fz2n_tiles

subroutine acc_fz2n_kernel(numat, mpack, f, ptot, qe, iorbs, nat, coord, nij, ntile, ntp, tile_n, spos, tstart, tlist, &
  part)
! Per atom: the s,p NDDO pair contributions to its diagonal block, summed from the partial sums of
! acc_fz2n_tiles over the atom's tile pairs in a fixed order, and the point-charge pairs.
  implicit none
  integer, intent(in) :: numat, mpack, ntile, ntp, tile_n
  double precision, intent(inout) :: f(mpack)
  double precision, intent(in) :: ptot(mpack), qe(numat), coord(3, numat)
  integer, intent(in) :: iorbs(numat), nat(numat), nij(numat, numat), spos(numat), tstart(ntile + 1), tlist(*)
  double precision, intent(in) :: part(10, tile_n, 2, *)
  integer :: a, b, k, d, na, x, ta, tp, sd, kl
  double precision :: c(10), cm, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, sm
  !$acc routine(acc_fz_far) seq
  !$acc data copyin(qe, coord, iorbs, nat) present(nij, spos, tstart, tlist, part) deviceptr(f, ptot)
  !$acc parallel loop gang vector_length(128) private(na, d, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, sm, x, ta, kl)
  do a = 1, numat
    na = iorbs(a)
    if (na == 0) cycle
    d = nij(a, a)
    s1 = 0.d0; s2 = 0.d0; s3 = 0.d0; s4 = 0.d0; s5 = 0.d0
    s6 = 0.d0; s7 = 0.d0; s8 = 0.d0; s9 = 0.d0; s10 = 0.d0; sm = 0.d0
! NDDO pairs, s,p atoms only (an atom with d orbitals has no entries in the tile partials)
    x = 1
    ta = 1
    kl = 0                                ! not an s,p atom: no tile entries
    if (spos(a) > 0) then
      x = (spos(a) - 1)/tile_n + 1
      ta = spos(a) - (x - 1)*tile_n
      kl = tstart(x + 1) - 1
    end if
    !$acc loop vector private(tp, sd) reduction(+:s1, s2, s3, s4, s5, s6, s7, s8, s9, s10)
    do k = tstart(x), kl
      tp = abs(tlist(k))
      sd = 1
      if (tlist(k) < 0) sd = 2
      s1 = s1 + part(1, ta, sd, tp); s2 = s2 + part(2, ta, sd, tp); s3 = s3 + part(3, ta, sd, tp)
      s4 = s4 + part(4, ta, sd, tp); s5 = s5 + part(5, ta, sd, tp); s6 = s6 + part(6, ta, sd, tp)
      s7 = s7 + part(7, ta, sd, tp); s8 = s8 + part(8, ta, sd, tp); s9 = s9 + part(9, ta, sd, tp)
      s10 = s10 + part(10, ta, sd, tp)
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

subroutine acc_fz_pair_sp(ii, jj, numat, mpack, f, ptot, iorbs, nat, coord, nij, ci, cj)
! NDDO pair (ii, jj), ii > jj, both atoms s or s,p: the integrals once, the contributions to the
! diagonal blocks of both atoms (ci for ii, cj for jj; formerly computed separately, with the
! integrals evaluated twice, for each atom), and the exchange term subtracted from the off-diagonal
! block (ii, jj) of f.
  implicit none
  !$acc routine seq
  !$acc routine(rotate_sp_w) seq
  integer, intent(in) :: ii, jj, numat, mpack
  double precision, intent(inout) :: f(mpack)
  double precision, intent(in) :: ptot(mpack), coord(3, numat)
  integer, intent(in) :: iorbs(numat), nat(numat), nij(numat, numat)
  double precision, intent(out) :: ci(10), cj(10)
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
  integer :: ia, ja, di, dj, l, i, j, k, m, jk, kj, li
  double precision :: w(100), p(16), sum, sumoff, sumdia
  call rotate_sp_w(nat(ii), nat(jj), coord(1, ii), coord(1, jj), w)   ! w only: the Fock build needs no core terms
  ia = iorbs(ii)
  ja = iorbs(jj)
  di = nij(ii, ii)
  dj = nij(jj, jj)
  l = nij(ii, jj)
  do i = 1, 10
    ci(i) = 0.d0
    cj(i) = 0.d0
  end do
  if (ia == 4 .and. ja == 4) then
! Heavy atom - heavy atom. ii: Coulomb from the density of jj, and the exchange block.
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
        ci(i) = ci(i) + p(j)*w(jjndx(i, j))
      end do
    end do
    do j = 1, 16
      sum = 0.d0
      do i = 1, 16
        sum = sum + ptot(l + i)*w(kkind(i, j))
      end do
      f(l + j) = f(l + j) - sum*0.5d0
    end do
! jj: Coulomb from the density of ii.
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
        cj(i) = cj(i) + p(j)*w(jindx(i, j))
      end do
    end do
  else if (ia == 4 .or. ja == 4) then
! Heavy atom - light atom. The heavy atom gets W (its 10 pairs) times the light atom's s density; the
! light atom gets the heavy atom's density contracted with W.
    if (ia == 4) then
      do k = 1, 10
        ci(k) = ptot(dj + 1)*w(k)
      end do
      li = di
    else
      do k = 1, 10
        cj(k) = ptot(di + 1)*w(k)
      end do
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
    if (ia == 4) then
      cj(1) = sumoff*2.d0 + sumdia
    else
      ci(1) = sumoff*2.d0 + sumdia
    end if
    do i = 1, 4
      sum = 0.d0
      do j = 1, 4
        sum = sum + ptot(l + j)*w((max(i, j)*(max(i, j) - 1))/2 + min(i, j))
      end do
      f(l + i) = f(l + i) - sum*0.5d0
    end do
  else
! Light atom - light atom
    ci(1) = ptot(dj + 1)*w(1)
    cj(1) = ptot(di + 1)*w(1)
    f(l + 1) = f(l + 1) - ptot(l + 1)*w(1)*0.5d0
  end if
end subroutine acc_fz_pair_sp

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
  write (iw, '(a,es10.3,a,i0,a,es10.3,a,f8.3,a)') ' MOPAC GPU Fock test: max |F(CPU) - F(GPU)| ', dmax, ' eV at ', imax, &
    ' (max |F| ', fmax, ');  GPU part ', tg, ' s'
  flush (iw)
end subroutine acc_fock_compare

subroutine acc_h_invalidate()
! Called at every entry to iter_for_MOZYME: h may have changed since the last SCF (new geometry,
! hcore_for_MOZYME, addhcr, add_more_interactions), so the device copy is refreshed on next use.
! The device copies of F and P are dropped too: between SCFs (geometry steps, LOCATE-TS, gradients)
! the host P is rewritten by code that does not clear pd_ok, and the sampled check in acc_fz2n
! cannot prove equality.
#ifdef _OPENACC
  use acc_mozyme_C, only : hd_ok, eimp_dev, fd_ok, pd_ok
  hd_ok = .false.
  eimp_dev = .false.
  fd_ok = .false.
  pd_ok = .false.
#endif
end subroutine acc_h_invalidate

subroutine acc_pin_begin()
! Called at every entry to iter_for_MOZYME: transfers of common f, p, h may pin them (acc_pin) until
! acc_pin_end, which every caller of iter_for_MOZYME calls after it returns.
  use acc_mozyme_C, only : pin_scope, acc_pin_release
  call acc_pin_release()
  pin_scope = .true.
end subroutine acc_pin_begin

subroutine acc_pin_end()
! Unpins the host arrays pinned during the last iter_for_MOZYME call (they may be freed after it).
  use acc_mozyme_C, only : acc_pin_release
  call acc_pin_release()
end subroutine acc_pin_end

logical function acc_eimp_skip()
! Called in place of the host eimp before diagg: .true. if the GPU diagg1 will run on the device copies
! of F and P left by the GPU Fock build; it then applies eimp to the device P itself
! (acc_eimp_kernel), and the host P keeps the density (no host reader needs the eimp values, except
! the CPU fallbacks of diagg1 and the .den write of PLS_faulty, which call acc_eimp_host_fallback).
#ifdef _OPENACC
  use MOZYME_C, only : lijbo
  use acc_mozyme_C, only : acc_mozyme_sync, allow_diagg, fd_ok, pd_ok, eimp_dev
#endif
  implicit none
#ifdef _OPENACC
  integer, external :: acc_diagg_mode
  logical :: dummy
  acc_eimp_skip = .false.
  eimp_dev = .false.
  if (.not. lijbo) return
  if (acc_diagg_mode() /= 1) return          ! off, or test mode (the CPU diagg1 also runs)
  dummy = acc_mozyme_sync()
  if (.not. (allow_diagg .and. fd_ok .and. pd_ok)) return
  eimp_dev = .true.
  acc_eimp_skip = .true.
#else
  acc_eimp_skip = .false.
#endif
end function acc_eimp_skip

subroutine acc_eimp_host_fallback()
! A host path needs the eimp values in the host P after acc_eimp_skip: compute them now.
#ifdef _OPENACC
  use acc_mozyme_C, only : eimp_dev
  if (eimp_dev) then
    call eimp()
    eimp_dev = .false.
  end if
#endif
end subroutine acc_eimp_host_fallback

logical function acc_helecz(ee)
! Electronic energy of MOZYME (helecz) from the device copies of F and P left by the GPU Fock build and
! a device copy of h: one gang per atom (its off-diagonal blocks with lower atoms, then its diagonal
! block), partial sums returned per atom and added on the host in atom order, so the result is
! reproducible. .false. (ee not set) if the device copies are not current: the caller uses helecz.
#ifdef _OPENACC
  use molkst_C, only : numat, mpack
  use MOZYME_C, only : iorbs, lijbo
  use common_arrays_C, only : h
  use acc_mozyme_C, only : acc_report, acc_mozyme_sync, acc_mozyme_hbuf, acc_to_device, acc_matches_device, nij, fd_c, pd_c, &
    hd_c, fd_ok, pd_ok, hd_ok
  use iso_c_binding, only : c_f_pointer
  use chanel_C, only : iw
#endif
  implicit none
  double precision, intent(out) :: ee
#ifdef _OPENACC
  integer, external :: acc_fock_mode
  double precision, external :: helecz
  double precision, pointer, contiguous :: fdev(:), pdev(:), hdev(:)
  double precision, allocatable :: eo(:), ed(:)
  logical :: dummy, check
  double precision :: eh
  integer :: i
  integer, save :: nstale = 0
  double precision :: so, sd
  acc_helecz = .false.
  if (acc_fock_mode() /= 1 .or. .not. lijbo) return
  dummy = acc_mozyme_sync()
  if (.not. (fd_ok .and. pd_ok) .or. .not. allocated(h)) return
  if (size(h) < mpack) return
  if (.not. acc_mozyme_hbuf(mpack)) return
  if (hd_ok) then
    if (.not. acc_matches_device(hd_c, h, mpack, 0.d0)) then
      nstale = nstale + 1
      if (nstale <= 20) write (iw, '(a)') ' MOPAC GPU energy: device copy of H was stale; copied again'
      hd_ok = .false.
    end if
  end if
  check = .not. hd_ok          ! first use in this SCF: also compute helecz on the host and compare
  if (.not. hd_ok) then
    call acc_to_device(hd_c, h, mpack)
    hd_ok = .true.
  end if
  call c_f_pointer(fd_c, fdev, [mpack])
  call c_f_pointer(pd_c, pdev, [mpack])
  call c_f_pointer(hd_c, hdev, [mpack])
  allocate (eo(numat), ed(numat))
  call acc_helecz_kernel(numat, mpack, fdev, pdev, hdev, nij, iorbs, eo, ed)
  so = 0.d0
  sd = 0.d0
  do i = 1, numat
    so = so + eo(i)
    sd = sd + ed(i)
  end do
  ee = so + 0.5d0*sd
  deallocate (eo, ed)
  acc_helecz = .true.
  if (check) then
    eh = helecz()
    if (acc_report()) write (iw, '(a,es24.16,a,es10.3,a)') ' MOPAC GPU energy: GPU ', ee, ' eV, |GPU - CPU| ', abs(ee - eh), ' eV'
    if (abs(ee - eh) > 1.d-9*max(1.d0, abs(eh))) write (iw, '(a)') &
      ' MOPAC GPU energy: WARNING: the GPU and CPU electronic energies differ by more than 1e-9 (relative)'
    flush (iw)
  end if
#else
  ee = 0.d0
  acc_helecz = .false.
#endif
end function acc_helecz

#ifdef _OPENACC
subroutine acc_helecz_kernel(numat, mpack, f, p, h, nij, iorbs, eo, ed)
! Per atom i: eo(i) = sum of p*(h+f) over the off-diagonal blocks (i, j < i) and the off-diagonal
! elements of the diagonal block; ed(i) = the same over the diagonal elements (helecz weights them 1/2).
  implicit none
  integer, intent(in) :: numat, mpack
  double precision, intent(in) :: f(mpack), p(mpack), h(mpack)
  integer, intent(in) :: nij(numat, numat), iorbs(numat)
  double precision, intent(out) :: eo(numat), ed(numat)
  integer :: i, j, k, m, l1, l2, nb
  double precision :: s, so, sd
  !$acc data present(nij) copyin(iorbs) copyout(eo, ed) deviceptr(f, p, h)
  !$acc parallel loop gang vector_length(128) private(s, so, sd, k)
  do i = 1, numat
    s = 0.d0
    !$acc loop vector reduction(+:s) private(k, nb)
    do j = 1, i - 1
      k = nij(j, i)                ! nijbo is symmetric; (j, i) is contiguous in j
      if (k >= 0) then
        nb = iorbs(i)*iorbs(j)
        do m = k + 1, k + nb
          s = s + p(m)*(h(m) + f(m))
        end do
      end if
    end do
    so = 0.d0
    sd = 0.d0
    k = nij(i, i)
    !$acc loop seq
    do l1 = 1, iorbs(i)
      !$acc loop seq
      do l2 = 1, l1 - 1
        k = k + 1
        so = so + p(k)*(h(k) + f(k))
      end do
      k = k + 1
      sd = sd + p(k)*(h(k) + f(k))
    end do
    eo(i) = s + so
    ed(i) = sd
  end do
  !$acc end data
end subroutine acc_helecz_kernel
#endif
