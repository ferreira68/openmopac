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

subroutine acc_fz2n(f, ptot, qe, iorbs, nat, coord)
#ifdef _OPENACC
  use molkst_C, only : numat, mpack, numcal
  use MOZYME_C, only : nijbo
  use parameters_C, only : dorbs, iod
#endif
  implicit none
  double precision, intent(inout) :: f(*)
  double precision, intent(in) :: ptot(*), qe(*), coord(3,*)
  integer, intent(in) :: iorbs(*), nat(*)
#ifdef _OPENACC
  integer, save :: key_numcal = -1, key_mpack = -1, key_numat = -1, nnear = 0
  integer, allocatable, save :: nstart(:), near(:), nij(:,:)
  logical :: spi, spj
  integer :: a, b, ii, jj, k
  if (numcal /= key_numcal .or. mpack /= key_mpack .or. numat /= key_numat) then
! New geometry or new set of interactions: rebuild the NDDO s,p neighbor lists (CSR) and
! refresh the device copies of nijbo and the parameters.
    if (allocated(nstart)) then
      !$acc exit data delete(nstart, near, nij)
      deallocate (nstart, near, nij)
    end if
    call acc_update_nddo()
    allocate (nstart(numat + 1), nij(numat, numat))
    nij = nijbo(1:numat, 1:numat)
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
    !$acc enter data copyin(nstart, near, nij)
    key_numcal = numcal
    key_mpack = mpack
    key_numat = numat
  end if
  call acc_fz2n_kernel(numat, mpack, f, ptot, qe, iorbs, nat, coord, nij, nstart, near, max(1, nnear))
#endif
end subroutine acc_fz2n

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
  !$acc data copy(f) copyin(ptot, qe, coord, iorbs, nat) present(nij, nstart, near)
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
