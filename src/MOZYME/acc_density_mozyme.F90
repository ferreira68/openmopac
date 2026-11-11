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

! OpenACC (GPU) version of density_for_MOZYME, full density (mode 0, lijbo):
!   P = 2 sum over occupied LMOs i of c_i c_i^T, on the atom-pair blocks of each LMO.
!
! One gang per LMO; its atom pairs in parallel. Many LMOs add to the same elements of P, so the
! additions are atomic. To make the result independent of the order of those additions (bitwise
! reproducible from run to run), each product is rounded to a 64-bit fixed-point number with
! scale 2**58 (resolution 3.5e-18) and accumulated with integer atomics, which are associative.
! For orthonormal LMOs every partial sum of an element is bounded by about 1 in magnitude
! (Cauchy-Schwarz), far inside the range of the representation (+-32).
!
! The result is left in the device buffer pd (pd_ok = .true.) and copied to the host p, so the
! next Fock build need not copy P to the device.
!
! Enabled by the environment variable MOPAC_ACC_DENS ("test": run both and compare).

integer function acc_density_mode()
  implicit none
  integer, save :: mode = -1
#ifdef _OPENACC
  character(len=16) :: env
  integer :: stat
  if (mode < 0) then
    call get_environment_variable("MOPAC_ACC_DENS", env, status=stat)
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
  acc_density_mode = mode
end function acc_density_mode

subroutine acc_density_core(p, nclose, resident)
! Full density matrix (mode 0) of the first nclose occupied LMOs into p(1:mpack).
! If resident, p is common_arrays p, and pd is marked as holding it.
#ifdef _OPENACC
  use molkst_C, only : numat, mpack
  use MOZYME_C, only : ncf, nncf, ncocc, icocc, cocc, iorbs
  use acc_mozyme_C, only : nij, acc_mozyme_sync, acc_mozyme_buffers, acc_from_device, pd_c, pd_ok, fd_ok
  use common_arrays_C, only : pcom => p
  use iso_c_binding, only : c_f_pointer, c_ptr, c_associated, c_null_ptr, c_long_long
#endif
  implicit none
  integer, intent(in) :: nclose
  double precision, intent(inout) :: p(*)
  logical, intent(in) :: resident
#ifdef _OPENACC
  interface
    type(c_ptr) function c_acc_malloc(bytes) bind(C, name="acc_malloc")
      import :: c_ptr, c_long_long
      integer(c_long_long), value :: bytes
    end function c_acc_malloc
    subroutine c_acc_free(dev) bind(C, name="acc_free")
      import :: c_ptr
      type(c_ptr), value :: dev
    end subroutine c_acc_free
  end interface
  double precision, pointer, contiguous :: pdev(:)
  integer(8), pointer, contiguous :: pacc(:)
  type(c_ptr), save :: acc_c = c_null_ptr
  integer, save :: acc_len = 0
  logical :: dummy
  dummy = acc_mozyme_sync()
  call acc_mozyme_buffers(mpack)
  if (mpack > acc_len) then
    if (c_associated(acc_c)) call c_acc_free(acc_c)
    acc_c = c_acc_malloc(int(mpack, c_long_long)*8_c_long_long)
    if (.not. c_associated(acc_c)) stop 'acc_density_core: acc_malloc failed'
    acc_len = mpack
  end if
  call c_f_pointer(pd_c, pdev, [mpack])
  call c_f_pointer(acc_c, pacc, [mpack])
  pd_ok = .false.
  fd_ok = .false.
  call acc_density_kernels(numat, mpack, nclose, size(icocc), size(cocc), nij, iorbs, ncf, nncf, ncocc, icocc, &
    cocc, pacc, pdev)
  call acc_from_device(p, pd_c, mpack)
  if (resident .and. allocated(pcom)) pd_ok = (loc(p(1)) == loc(pcom(1)))
#endif
end subroutine acc_density_core

#ifdef _OPENACC
subroutine acc_density_kernels(numat, mpack, nclose, licocc, lcocc, nij, iorbs, ncf, nncf, ncocc, icocc, cocc, pacc, p)
  implicit none
  integer, intent(in) :: numat, mpack, nclose, licocc, lcocc
  integer, intent(in) :: nij(numat, numat), iorbs(numat), ncf(nclose), nncf(nclose), ncocc(nclose), icocc(licocc)
  double precision, intent(in) :: cocc(lcocc)
  integer(8), intent(inout) :: pacc(mpack)
  double precision, intent(inout) :: p(mpack)
  double precision, parameter :: scale = 2.d0**58, unscale = 2.d0**(-58)
  integer, allocatable :: offo(:)
  integer :: i, jj, kk, j, k, l, j1, k1, nj, ja, ka, n0, nf, loop, m
  integer(8) :: v
  double precision :: s
  allocate (offo(licocc))
  !$acc data present(nij) copyin(iorbs, ncf(1:nclose), nncf(1:nclose), ncocc(1:nclose), icocc, cocc) create(offo) &
  !$acc& deviceptr(pacc, p)
  !$acc parallel loop gang vector
  do m = 1, mpack
    pacc(m) = 0_8
  end do
  !$acc parallel loop gang vector_length(128) private(n0, nf, loop, l)
  do i = 1, nclose
    n0 = nncf(i)
    nf = ncf(i)
    loop = ncocc(i)
    l = 0
    !$acc loop seq
    do jj = 1, nf
      offo(n0 + jj) = l
      l = l + iorbs(icocc(n0 + jj))
    end do
    !$acc loop vector collapse(2) private(j, k, nj, ja, ka, l, j1, k1, s, v)
    do jj = 1, nf
      do kk = 1, nf
        j = icocc(n0 + jj)
        k = icocc(n0 + kk)
        nj = iorbs(j)
        ja = offo(n0 + jj)
        ka = loop + offo(n0 + kk)
        if (j == k) then
          l = nij(j, k)
          do j1 = 1, nj
            s = cocc(ja + j1 + loop)
            do k1 = 1, j1
              l = l + 1
              v = nint(cocc(ka + k1)*s*scale, 8)
              !$acc atomic update
              pacc(l) = pacc(l) + v
            end do
          end do
        else if (j > k) then
          l = nij(j, k)
          if (l >= 0) then
            do j1 = 1, nj
              s = cocc(ja + j1 + loop)
              do k1 = 1, iorbs(k)
                l = l + 1
                v = nint(cocc(ka + k1)*s*scale, 8)
                !$acc atomic update
                pacc(l) = pacc(l) + v
              end do
            end do
          end if
        end if
      end do
    end do
  end do
  !$acc parallel loop gang vector
  do m = 1, mpack
    p(m) = 2.d0*(dble(pacc(m))*unscale)
  end do
  !$acc end data
  deallocate (offo)
end subroutine acc_density_kernels
#endif

subroutine acc_density_compare(p, pg, tg)
! MOPAC_ACC_DENS=test: largest difference between the CPU (p) and GPU (pg) density matrices.
  use molkst_C, only : mpack
  use chanel_C, only : iw
  implicit none
  double precision, intent(in) :: p(mpack), pg(mpack), tg
  integer :: i
  double precision :: d
  d = 0.d0
  do i = 1, mpack
    d = max(d, abs(p(i) - pg(i)))
  end do
  write (iw, '(a,es10.3,a,f8.3,a)') ' ACC DENS TEST: max |P(CPU) - P(GPU)| ', d, ';  GPU ', tg, ' s'
  flush (iw)
end subroutine acc_density_compare
