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

! Two-center integrals for atom pairs in which neither atom has d orbitals (s or s,p basis).
!
! These routines are copies of rotate, rotatd, rotmat, tx and elenuc restricted to that case,
! with every array sized for s and p functions (at most 10 one-center pairs per atom, 100
! two-electron integrals per atom pair) instead of the worst case with d orbitals (45, 2025).
! The arithmetic and its order are unchanged, so for these pairs the results are bitwise
! identical to rotate. Their purpose is the GPU (OpenACC) integral kernel: the general routines'
! worst-case scratch (about 71 KB per thread) limits occupancy and turns the kernel into
! local-memory traffic.
!
! Preconditions (checked by the callers): .not. (dorbs(ni) .or. dorbs(nj)), iod(ni) = iod(nj) = 0,
! and a molecular system (id = 0).

  subroutine rotate_sp(ni, nj, xi, xj, w, e1b, e2a, enuc)
  use parameters_C, only : natorb
  !$acc routine seq
  !$acc routine(rotatd_sp) seq
  !$acc routine(elenuc_sp) seq
  implicit none
  integer, intent(in) :: ni, nj
  double precision, intent(in) :: xi(3), xj(3)
  double precision, intent(out) :: w(100), e1b(10), e2a(10), enuc
  integer :: i, li, lj, ik, k
  double precision :: rij, x(3), en(36)
  double precision :: sp(3,3), pp(6,3,3), cored(10,2)
  x(1) = xi(1) - xj(1)
  x(2) = xi(2) - xj(2)
  x(3) = xi(3) - xj(3)
  rij = x(1)*x(1) + x(2)*x(2) + x(3)*x(3)
  if (rij < 0.00002D0) then
    e1b = 0.D0
    e2a = 0.D0
    w = 0.D0
    enuc = 0.D0
    return
  end if
  li = natorb(ni)
  lj = natorb(nj)
  call rotatd_sp(ni, nj, xi, xj, w, enuc, sp, pp, cored)
  en(:) = 0.d0
  call elenuc_sp(1, li, li + 1, li + lj, en, sp, pp, cored)
  ik = 0
  do i = 1, li
    do k = 1, i
      ik = ik + 1
      e1b(ik) = en((i*(i - 1))/2 + k)
    end do
  end do
  ik = 0
  do i = li + 1, li + lj
    do k = li + 1, i
      ik = ik + 1
      e2a(ik) = en((i*(i - 1))/2 + k)
    end do
  end do
  end subroutine rotate_sp

  subroutine rotatd_sp(ni, nj, ci, cj, w, enuc, sp, pp, cored)
  use funcon_C, only : a0, ev
  use mndod_C, only : indx, indexd
  use parameters_C, only : natorb, tore
  use molkst_C, only : l_feather
  !$acc routine seq
  !$acc routine(ccrep) seq
  !$acc routine(reppd) seq
  !$acc routine(spcore) seq
  !$acc routine(to_point) seq
  implicit none
  integer, intent(in) :: ni, nj
  double precision, intent(in) :: ci(3), cj(3)
  double precision, intent(out) :: w(100), enuc
  double precision, intent(out) :: sp(3,3), pp(6,3,3), cored(10,2)
  integer, dimension(45) :: met
  integer, dimension(34) :: ipos
  integer :: limkl, kl, ii, kk, limij, istep, i, i1, j1, ij, jj, mm, k, l, iw, iminus, j, jw
  double precision :: v(25,10), ri(22), rep(34), ww(100)
  double precision :: r, wrepp, cc, gab, rij, const, point
  logical :: logv(25,10)
  data met/ 1, 2, 3, 2, 3, 3, 2, 3, 3, 3, 4, 5, 5, 5, 6, 4, 5, 5, 5, 6, 6, &
    4, 5, 5, 5, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 4, 5, 5, 5, 5*6/
  data ipos/ 1, 5, 11, 12, 12, 2, 6, 13, 14, 14, 3, 8, 16, 18, 18, 7, 15, &
    10, 20, 4, 9, 17, 19, 21, 7, 15, 10, 20, 22, 4, 9, 17, 21, 19/
  call rotmat_sp(nj, ni, ci, cj, r, sp, pp)
  call reppd (ni, nj, r, ri, gab)
  rij = r
  r = r/a0
  call spcore (ni, nj, r, cored)
  if (l_feather) then
    call to_point(rij, point, const)
  else
    const = 1.d0
    point = 0.d0
  end if
  rep(:34) = ri(ipos)          ! reppd2 without d orbitals
  point = -(eV/r)*tore(nj)
  cored(1,1) = cored(1,1)*const + (1.d0 - const)*point
  cored(2,1) = cored(2,1)*const
  cored(3,1) = cored(3,1)*const + (1.d0 - const)*point
  cored(4,1) = cored(4,1)*const + (1.d0 - const)*point
  cored(5,1) = cored(5,1)*const
  cored(6,1) = cored(6,1)*const
  cored(7,1) = cored(7,1)*const + (1.d0 - const)*point
  cored(8,1) = cored(8,1)*const
  cored(9,1) = cored(9,1)*const + (1.d0 - const)*point
  cored(10,1) = cored(10,1)*const + (1.d0 - const)*point
  point = -(eV/r)*tore(ni)
  cored(1,2) = cored(1,2)*const + (1.d0 - const)*point
  cored(2,2) = cored(2,2)*const
  cored(3,2) = cored(3,2)*const + (1.d0 - const)*point
  cored(4,2) = cored(4,2)*const + (1.d0 - const)*point
  cored(5,2) = cored(5,2)*const
  cored(6,2) = cored(6,2)*const
  cored(7,2) = cored(7,2)*const + (1.d0 - const)*point
  cored(8,2) = cored(8,2)*const
  cored(9,2) = cored(9,2)*const + (1.d0 - const)*point
  cored(10,2) = cored(10,2)*const + (1.d0 - const)*point
  ii = natorb(ni)
  kk = natorb(nj)
  ww = 0
  if (ii*kk > 0) then
    limij = indx(ii,ii)
    limkl = indx(kk,kk)
    istep = limkl*limij
    ww(:istep) = 0.0D0
    call tx_sp (ii, kk, rep, logv, v, sp, pp)
    do i1 = 1, ii
      do j1 = 1, i1
        ij = indexd(i1,j1)
        jj = indx(i1,j1)
        mm = met(jj)
        do k = 1, kk
          do l = 1, k
            kl = indx(k,l)
            if (.not.logv(ij,kl)) cycle
            wrepp = v(ij,kl)
            select case (mm)
            case (1)
              iw = ((indx(1,1)-1)*limkl + kl)
              ww(iw) = wrepp
            case (2)
              do i = 1, 3
                iw = ((indx(i + 1,1)-1)*limkl + kl)
                ww(iw) = ww(iw) + sp(i1-1,i)*wrepp
              end do
            case (3)
              do i = 1, 3
                cc = pp(i,i1-1,j1-1)
                iw = ((indx(i + 1,i + 1)-1)*limkl + kl)
                ww(iw) = ww(iw) + cc*wrepp
                iminus = i - 1
                if (iminus == 0) cycle
                do j = 1, iminus
                  cc = pp(1+i+j,i1-1,j1-1)
                  iw = ((indx(i + 1,j + 1)-1)*limkl + kl)
                  ww(iw) = ww(iw) + cc*wrepp
                end do
              end do
            end select
          end do
        end do
      end do
    end do
  end if
  iw = (ii*(ii + 1))/2
  jw = (kk*(kk + 1))/2
  k = 0
  do ij = 1, iw                ! w2mat
    do kl = 1, jw
      k = k + 1
      w(k) = ww((ij - 1)*jw + kl)
    end do
  end do
  call ccrep (ni, nj, r, enuc, gab)
  end subroutine rotatd_sp

  subroutine rotmat_sp(nj, ni, coordi, coordj, r, sp, pp)
  !$acc routine seq
  implicit none
  integer, intent(in) :: nj, ni
  double precision, intent(out) :: r
  double precision, intent(in) :: coordi(3), coordj(3)
  double precision, intent(out) :: sp(3,3), pp(6,3,3)
  double precision, parameter :: small = 1.0D-07
  integer :: k
  double precision :: p(3,3)
  double precision :: x11, x22, x33, b, sqb, sb, ca, sa, cb
  x11 = coordj(1) - coordi(1)
  x22 = coordj(2) - coordi(2)
  x33 = coordj(3) - coordi(3)
  b = x11*x11 + x22*x22
  r = sqrt(b + x33*x33)
  sqb = sqrt(b)
  sb = sqb/r
  if (sb > small) then
    ca = x11/sqb
    sa = x22/sqb
    cb = x33/r
  else
    sa = 0.d0
    sb = 0.d0
    if (x33 < 0.d0) then
      ca = -1.d0
      cb = -1.d0
    else if (x33 > 0.d0) then
      ca = 1.d0
      cb = 1.d0
    else
      ca = 0.d0
      cb = 0.d0
    end if
  end if
  p(1,1) = ca*sb
  p(2,1) = ca*cb
  p(3,1) = -sa
  p(1,2) = sa*sb
  p(2,2) = sa*cb
  p(3,2) = ca
  p(1,3) = cb
  p(2,3) = -sb
  p(3,3) = 0.d0
  sp = p
  do k = 1, 3
    pp(1,k,k) = p(k,1)*p(k,1)
    pp(2,k,k) = p(k,2)*p(k,2)
    pp(3,k,k) = p(k,3)*p(k,3)
    pp(4,k,k) = p(k,1)*p(k,2)
    pp(5,k,k) = p(k,1)*p(k,3)
    pp(6,k,k) = p(k,2)*p(k,3)
    if (k == 1) cycle
    pp(1,k,:k-1) = 2.D0*p(k,1)*p(:k-1,1)
    pp(2,k,:k-1) = 2.D0*p(k,2)*p(:k-1,2)
    pp(3,k,:k-1) = 2.D0*p(k,3)*p(:k-1,3)
    pp(4,k,:k-1) = p(k,1)*p(:k-1,2) + p(k,2)*p(:k-1,1)
    pp(5,k,:k-1) = p(k,1)*p(:k-1,3) + p(k,3)*p(:k-1,1)
    pp(6,k,:k-1) = p(k,2)*p(:k-1,3) + p(k,3)*p(:k-1,2)
  end do
  if (.false.) r = r + dble(ni + nj)   ! ni, nj kept for the same interface as rotmat
  end subroutine rotmat_sp

  subroutine tx_sp(ii, kk, rep, logv, v, sp, pp)
  use mndod_C, only : indexd, indx, ind2
  !$acc routine seq
  implicit none
  integer, intent(in) :: ii, kk
  double precision, intent(in) :: rep(34)
! Rows are indexed by indexd (9x9 column-wise lower triangle: 1-4, 10-12, 18-19, 25 for s,p),
! columns by indx (packed, at most 10 for s,p).
  double precision, intent(out) :: v(25,10)
  logical, intent(out) :: logv(25,10)
  double precision, intent(in) :: sp(3,3), pp(6,3,3)
  integer, dimension(45) :: met
  integer :: limkl, k, i1, j1, ij, k1, l1, kl, nd, ll, mm, l
  double precision :: wrepp
  data met/ 1, 2, 3, 2, 3, 3, 2, 3, 3, 3, 4, 5, 5, 5, 6, 4, 5, 5, 5, 6, 6, &
    4, 5, 5, 5, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 4, 5, 5, 5, 5*6/
  limkl = indx(kk,kk)
  logv(:,:limkl) = .FALSE.
  v(:,:limkl) = 0.D0
  do i1 = 1, ii
    do j1 = 1, i1
      ij = indexd(i1,j1)
      do k1 = 1, kk
        do l1 = 1, k1
          kl = indexd(k1,l1)
          nd = ind2(ij,kl)
          if (nd == 0) cycle
          wrepp = rep(nd)
          ll = indx(k1,l1)
          mm = met(ll)
          select case (mm)
          case (1)
            v(ij,1) = wrepp
          case (2)
            k = k1 - 1
            v(ij,2) = v(ij,2) + sp(k,1)*wrepp
            v(ij,4) = v(ij,4) + sp(k,2)*wrepp
            v(ij,7) = v(ij,7) + sp(k,3)*wrepp
          case (3)
            k = k1 - 1
            l = l1 - 1
            v(ij,3) = v(ij,3) + pp(1,k,l)*wrepp
            v(ij,6) = v(ij,6) + pp(2,k,l)*wrepp
            v(ij,10) = v(ij,10) + pp(3,k,l)*wrepp
            v(ij,5) = v(ij,5) + pp(4,k,l)*wrepp
            v(ij,8) = v(ij,8) + pp(5,k,l)*wrepp
            v(ij,9) = v(ij,9) + pp(6,k,l)*wrepp
            cycle
          end select
        end do
      end do
      where (v(ij,:limkl) /= 0.00D00)
        logv(ij,:limkl) = .TRUE.
      end where
    end do
  end do
  end subroutine tx_sp

  subroutine elenuc_sp(ia, ib, ja, jb, h, sp, pp, cored)
  use mndod_C, only : indpp
  !$acc routine seq
  implicit none
  integer, intent(in) :: ia, ib, ja, jb
  double precision, intent(inout) :: h(36)
  double precision, intent(in) :: sp(3,3), pp(6,3,3), cored(10,2)
  integer :: k, l, n, i, ind1, j, ind2, m, ipp
  do n = 1, 2
    if (n == 1) then
      k = ia
      l = ib
    else
      k = ja
      l = jb
    end if
    do i = k, l
      ind1 = i - k
      do j = k, i
        ind2 = j - k
        m = (i*(i - 1))/2 + j
        if (ind1 == 0) then
          h(m) = h(m) + cored(1,n)
        else if (ind2 == 0) then
          h(m) = h(m) + sp(1,ind1)*cored(2,n)
        else
          ipp = indpp(ind1,ind2)
          h(m) = h(m) + cored(3,n)*pp(ipp,1,1) + cored(4,n)*(pp(ipp,2,2)+pp(ipp,3,3))
        end if
      end do
    end do
  end do
  end subroutine elenuc_sp
