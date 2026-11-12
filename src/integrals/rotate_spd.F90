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

! Two-center integrals for atom pairs in which exactly one atom has d orbitals (an s,p,d atom with
! an s or s,p atom).
!
! rotate_spd, rotatd_spd and tx_spd are copies of rotate, rotatd and tx (mndod.F90) restricted to
! that case, with the scratch sized for it: at most 45 x 10 one-center pairs, so 450 two-electron
! integrals per atom pair instead of the d,d worst case of 2025. The arithmetic and its order are
! unchanged, so for these pairs the results are bitwise identical to rotate. Differences from the
! originals, none of which changes a result:
! - rotatd_spd accumulates the integrals in w directly instead of in a local ww that w2mat then
!   copied to w in the same (linear) order;
! - the rows of v and logv are indexed by indx(i1,j1) (at most 10 for an s,p atom, 45 for an
!   s,p,d atom) instead of indexd(i1,j1) (up to 25 for an s,p atom), so that v has at most
!   450 elements; ind2 is still addressed with indexd;
! - rotatd's unused checksum of ww and its call of elenuc (whose result it discarded; rotate calls
!   elenuc itself) are omitted.
! Their purpose is the GPU (OpenACC) Fock build: the per-thread stack of the general rotate
! (about 71 KB) is reserved by CUDA for every thread slot of the device, which costs gigabytes.
!
! Preconditions (checked by the callers): natorb = 9 for one atom, and for the other natorb <= 4,
! .not. dorbs and iod = 0; a molecular system (id = 0).

  subroutine rotate_spd(ni, nj, xi, xj, w, e1b, e2a, enuc)
  use parameters_C, only : natorb
  !$acc routine seq
  !$acc routine(rotatd_spd) seq
  !$acc routine(elenuc) seq
  implicit none
  integer, intent(in) :: ni, nj
  double precision, intent(in) :: xi(3), xj(3)
  double precision, intent(out) :: w(450), e1b(45), e2a(45), enuc
  integer :: i, li, lj, ik, k
  double precision :: rij, x(3), en(91)
  double precision :: sp(3,3), sd(5,5), pp(6,3,3), dp(15,5,3), d_d(15,5,5), cored(10,2)
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
  call rotatd_spd(ni, nj, xi, xj, w, enuc, sp, pp, sd, dp, d_d, cored)
  en(:) = 0.d0
  call elenuc (1, li, li + 1, li + lj, en, sp, sd, pp, dp, d_d, cored)
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
  end subroutine rotate_spd

      subroutine rotatd_spd(ni, nj, ci, cj, w, enuc, sp, pp, sd, dp, d_d, cored)
!-----------------------------------------------
!   M o d u l e s
!-----------------------------------------------
      use funcon_C, only : a0, ev
      use mndod_C, only : indx, indexd, inddd
      use parameters_C, only : natorb, iod, tore
      use molkst_C, only : l_feather, method_PM7
!     *
!     CALCULATION OF TWO-CENTER TWO-ELECTRON INTEGRALS
!     IN THE MOLECULAR COODINATE SYSTEM BY 2.d0-STEP PROCEDURE
!     *
!***********************************************************************
!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------
      !$acc routine seq
      !$acc routine(ccrep) seq
      !$acc routine(rotmat) seq
      !$acc routine(reppd) seq
      !$acc routine(spcore) seq
      !$acc routine(to_point) seq
      !$acc routine(reppd2) seq
      !$acc routine(tx_spd) seq
      implicit none
      integer, intent(in) :: ni, nj
!-----------------------------------------------
!   G l o b a l   P a r a m e t e r s
!-----------------------------------------------
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------
      double precision  :: ci(3)
      double precision  :: cj(3)
      double precision, intent(out) :: w(450)
      double precision  :: enuc
! Per-pair scratch owned by the caller (rotate), formerly module globals in mndod_C, so that atom
! pairs can be evaluated independently (threads, GPU).
      double precision, intent(inout) :: sp(3,3), sd(5,5), pp(6,3,3), dp(15,5,3), d_d(15,5,5)
      double precision, intent(inout) :: cored(10,2)
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------
      integer , dimension(45) :: met
      integer :: limkl, kl, ii, kk, limij, istep, i, i1, j1, ij, jj, mm, k, l, &
        iw, iminus, j, ij1, jw, li, lj
      double precision, dimension(450) :: v
      double precision, dimension(22) :: ri
      double precision, dimension(491) :: rep
      double precision :: r, wrepp, cc, gab, sum, rij, const, point
      logical, dimension(450) :: logv
!-----------------------------------------------
!
      data met/ 1, 2, 3, 2, 3, 3, 2, 3, 3, 3, 4, 5, 5, 5, 6, 4, 5, 5, 5, 6, 6, &
        4, 5, 5, 5, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 4, 5, 5, 5, 5*6/
!
!
      call rotmat (nj, ni, ci, cj, r, sp, pp, sd, dp, d_d)
!
      call reppd (ni, nj, r, ri, gab) ! Work out the 22 s-p terms
      rij = r
      r = r/a0
      call spcore (ni, nj, r, cored)
      if (l_feather) then
        call to_point(rij, point, const)
      else
        const = 1.d0
        point = 0.d0
      end if
      call reppd2 (ni, nj, r, ri, rep, cored) ! add in "d" terms
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

!
      ii = natorb(ni)
      kk = natorb(nj)
      w = 0
      if (ii*kk > 0) then
        limij = indx(ii,ii)
        limkl = indx(kk,kk)
        istep = limkl*limij
        w(:istep) = 0.0D0
  !
        call tx_spd (ii, kk, limij, rep, logv, v, sp, sd, pp, dp, d_d)
  !
        do i1 = 1, ii
          do j1 = 1, i1
  !
            ij = indexd(i1,j1)
            jj = indx(i1,j1)
            mm = met(jj)
  !
            do k = 1, kk
              do l = 1, k
                kl = indx(k,l)
                if (.not.logv(jj + (kl - 1)*limij)) cycle
                wrepp = v(jj + (kl - 1)*limij)
  !     GO TO (1,2,3,4,5,6),MM
  !
                select case (mm)
                case (1)
                  iw = ((indx(1,1)-1)*limkl + kl)
                  w(iw) = wrepp
                case (2)
                  do i = 1, 3
                    iw = ((indx(i + 1,1)-1)*limkl + kl)
                    w(iw) = w(iw) + sp(i1-1,i)*wrepp
                  end do
                case (3)
                  do i = 1, 3
                    cc = pp(i,i1-1,j1-1)
                    iw = ((indx(i + 1,i + 1)-1)*limkl + kl)
                    w(iw) = w(iw) + cc*wrepp
                    iminus = i - 1
                    if (iminus == 0) cycle
                    do j = 1, iminus
                      cc = pp(1+i+j,i1-1,j1-1)
                      iw = ((indx(i + 1,j + 1)-1)*limkl + kl)
                      w(iw) = w(iw) + cc*wrepp
                    end do
                  end do
                case (4)
                  do i = 1, 5
                    iw = ((indx(i + 4,1)-1)*limkl + kl)
                    w(iw) = w(iw) + sd(i1-4,i)*wrepp
                  end do
                case (5)
                  do i = 1, 5
                    do j = 1, 3
                      iw = ((indx(i + 4,j + 1)-1)*limkl + kl)
                      ij1 = 3*(i - 1) + j
                      w(iw) = w(iw) + dp(ij1,i1-4,j1-1)*wrepp
                    end do
                  end do
                case (6)
                  do i = 1, 5
                    cc = d_d(i,i1-4,j1-4)
                    iw = ((indx(i + 4,i + 4)-1)*limkl + kl)
                    w(iw) = w(iw) + cc*wrepp
                    iminus = i - 1
                    if (iminus == 0) cycle
                    do j = 1, iminus
                      ij1 = inddd(i,j)
                      cc = d_d(ij1,i1-4,j1-4)
                      iw = ((indx(i + 4,j + 4)-1)*limkl + kl)
                      w(iw) = w(iw) + cc*wrepp
                    end do
                  end do
                end select
               end do
            end do
          end do
        end do
      end if
      if (method_PM7) then
!
!  Elements with "d"-electrons have an imbalance between the nuclear-nuclear term and the other
!  two: electron-nuclear and electron-electron. To correct this specific error, the average value
!  of the two-electron two-center "d" coulomb integrals is set equal to the <ss|ss> integral.
! (The <ss|ss> integral is used in the nuclear-nuclear term)
!
        if (iod(ni) > 0) then
          sum = 0
          if (natorb(nj) == 9) then
              k = 45
            else if (natorb(nj) == 4) then
              k = 10
            else
              k = 1
            end if
          if (natorb(nj) > 1) then
            do i = 5,9                      !  "p" on nj with "d" on ni
              j = k*((i*(i + 1))/2 - 1)
              sum = sum + w(j + 3) + w(j + 6) + w(j + 10)
            end do
            sum = (w(1) - sum/15.d0)
            do i = 5,9
              j = k*((i*(i + 1))/2 - 1)
              do l = 2,4
                w((l*(l+1))/2 + j) = w((l*(l+1))/2 + j) + sum
              end do
            end do
            sum = cored(1,1) - (cored(3,1) + 2.d0*cored(4,1))/3.d0
            cored(3,1) = cored(3,1) + sum
            cored(4,1) = cored(4,1) + sum
          end if
          sum = 0.d0
          do i = 5,9
              sum = sum +w(k*((i*(i + 1))/2 - 1) + 1)
          end do
          sum = (w(1) - sum/5.d0)
          do i = 5,9                        !  "s" on nj with "d" on ni
            w(k*((i*(i + 1))/2 - 1) + 1) = w(k*((i*(i + 1))/2 - 1) + 1) + sum
          end do
          sum = cored(1,1) - (cored(7,1) + 2.d0*cored(9,1) + 2.d0*cored(10,1))/5.d0
          cored( 7,1) = cored( 7,1) + sum
          cored( 9,1) = cored( 9,1) + sum
          cored(10,1) = cored(10,1) + sum
        end if
        if (iod(nj) > 0) then
          sum = 0.d0
          if (iod(ni) > 0) then
            do i = 5, 9                     !  "d" with "d"  -  common to both atoms
              j = 45*((i*(i + 1))/2 - 1)
              sum = sum + (w(j + 15) + w(j + 21) + w(j + 28) + w(j + 36) + w(j + 45))
            end do
            sum = (w(1) - sum/25.d0)
            do i = 5, 9
              j = 45*((i*(i + 1))/2 - 1)
              do k = 5,9
                w((k*(k+1))/2 + j) = w((k*(k+1))/2 + j) + sum
              end do
            end do
          end if
          if (natorb(ni) > 1) then
            sum = 0.d0
            do i = 2,4                      !  "p" on ni with "d" on nj
              j = 45*((i*(i + 1))/2 - 1)
              sum = sum + (w(j + 15) + w(j + 21) + w(j + 28) + w(j + 36) + w(j + 45))
            end do
            sum = (w(1) - sum/15.d0)

            do i = 2,4
              j = 45*((i*(i + 1))/2 - 1)
              do k = 5,9
                w((k*(k+1))/2 + j) = w((k*(k+1))/2 + j) + sum
              end do
            end do
            sum = cored(1,2) - (cored(3,2) + 2.d0*cored(4,2))/3.d0
            cored(3,2) = cored(3,2) + sum
            cored(4,2) = cored(4,2) + sum
          end if
          sum = (w(1) - (w(15) + w(21) + w(28) + w(36) + w(45))/5.d0)
          do k = 5,9                        !  "s" on ni with "d" on nj
            w((k*(k+1))/2) = w((k*(k+1))/2) + sum
          end do
          sum = (cored(1,2) - (cored(7,2) + 2.d0*cored(9,2) + 2.d0*cored(10,2))/5.d0)
          cored( 7,2) = cored( 7,2) + sum
          cored( 9,2) = cored( 9,2) + sum
          cored(10,2) = cored(10,2) + sum
        end if
      end if
! *** CORE-CORE REPULSIONS FOR MNDO.
      call ccrep (ni, nj, r, enuc, gab)
      return
      end subroutine rotatd_spd


      subroutine tx_spd(ii, kk, ldv, rep, logv, v, sp, sd, pp, dp, d_d)
!-----------------------------------------------
!   M o d u l e s
!-----------------------------------------------
      use mndod_C, only : indexd, indx, ind2
!     *
!     ROTATION OF TWO-ELECTRON TWO-CENTER INTEGRALS IN SPD BASIS
!     FIRST STEP
!     *
      !$acc routine seq
      implicit none
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------
      integer , intent(in) :: ii
      integer , intent(in) :: kk
      integer , intent(in) :: ldv
      double precision , intent(in) :: rep(491)
      double precision , intent(out) :: v(ldv,*)
      logical , intent(out) :: logv(ldv,*)
      double precision, intent(in) :: sp(3,3), sd(5,5), pp(6,3,3), dp(15,5,3), d_d(15,5,5)
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------
      integer , dimension(45) :: met
      integer :: limkl, k, i1, j1, ij, k1, l1, kl, nd, ll, mm, l, ir
      double precision :: wrepp
!-----------------------------------------------
      data met/ 1, 2, 3, 2, 3, 3, 2, 3, 3, 3, 4, 5, 5, 5, 6, 4, 5, 5, 5, 6, 6, &
        4, 5, 5, 5, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 4, 5, 5, 5, 5*6/
!
      limkl = indx(kk,kk)
      logv(1:ldv,1:limkl) = .FALSE.
      v(1:ldv,1:limkl) = 0.D0
!
      do i1 = 1, ii
        do j1 = 1, i1
          ij = indexd(i1,j1)
          ir = indx(i1,j1)
!
          do k1 = 1, kk
!
            do l1 = 1, k1
              kl = indexd(k1,l1)
              nd = ind2(ij,kl)
              if (nd == 0) cycle
!
              wrepp = rep(nd)
              ll = indx(k1,l1)
              mm = met(ll)
!
              select case (mm)
              case (1)
                v(ir,1) = wrepp
              case (2)
                k = k1 - 1
                v(ir,2) = v(ir,2) + sp(k,1)*wrepp
                v(ir,4) = v(ir,4) + sp(k,2)*wrepp
                v(ir,7) = v(ir,7) + sp(k,3)*wrepp
              case (3)
                k = k1 - 1
                l = l1 - 1
                v(ir,3) = v(ir,3) + pp(1,k,l)*wrepp
                v(ir,6) = v(ir,6) + pp(2,k,l)*wrepp
                v(ir,10) = v(ir,10) + pp(3,k,l)*wrepp
                v(ir,5) = v(ir,5) + pp(4,k,l)*wrepp
                v(ir,8) = v(ir,8) + pp(5,k,l)*wrepp
                v(ir,9) = v(ir,9) + pp(6,k,l)*wrepp
                cycle
              case (4)
                k = k1 - 4
                v(ir,11) = v(ir,11) + sd(k,1)*wrepp
                v(ir,16) = v(ir,16) + sd(k,2)*wrepp
                v(ir,22) = v(ir,22) + sd(k,3)*wrepp
                v(ir,29) = v(ir,29) + sd(k,4)*wrepp
                v(ir,37) = v(ir,37) + sd(k,5)*wrepp
              case (5)
                k = k1 - 4
                l = l1 - 1
                v(ir,12) = v(ir,12) + dp(1,k,l)*wrepp
                v(ir,13) = v(ir,13) + dp(2,k,l)*wrepp
                v(ir,14) = v(ir,14) + dp(3,k,l)*wrepp
                v(ir,17) = v(ir,17) + dp(4,k,l)*wrepp
                v(ir,18) = v(ir,18) + dp(5,k,l)*wrepp
                v(ir,19) = v(ir,19) + dp(6,k,l)*wrepp
                v(ir,23) = v(ir,23) + dp(7,k,l)*wrepp
                v(ir,24) = v(ir,24) + dp(8,k,l)*wrepp
                v(ir,25) = v(ir,25) + dp(9,k,l)*wrepp
                v(ir,30) = v(ir,30) + dp(10,k,l)*wrepp
                v(ir,31) = v(ir,31) + dp(11,k,l)*wrepp
                v(ir,32) = v(ir,32) + dp(12,k,l)*wrepp
                v(ir,38) = v(ir,38) + dp(13,k,l)*wrepp
                v(ir,39) = v(ir,39) + dp(14,k,l)*wrepp
                v(ir,40) = v(ir,40) + dp(15,k,l)*wrepp
              case (6)
                k = k1 - 4
                l = l1 - 4
                v(ir,15) = v(ir,15) + d_d(1,k,l)*wrepp
                v(ir,21) = v(ir,21) + d_d(2,k,l)*wrepp
                v(ir,28) = v(ir,28) + d_d(3,k,l)*wrepp
                v(ir,36) = v(ir,36) + d_d(4,k,l)*wrepp
                v(ir,45) = v(ir,45) + d_d(5,k,l)*wrepp
                v(ir,20) = v(ir,20) + d_d(6,k,l)*wrepp
                v(ir,26) = v(ir,26) + d_d(7,k,l)*wrepp
                v(ir,27) = v(ir,27) + d_d(8,k,l)*wrepp
                v(ir,33) = v(ir,33) + d_d(9,k,l)*wrepp
                v(ir,34) = v(ir,34) + d_d(10,k,l)*wrepp
                v(ir,35) = v(ir,35) + d_d(11,k,l)*wrepp
                v(ir,41) = v(ir,41) + d_d(12,k,l)*wrepp
                v(ir,42) = v(ir,42) + d_d(13,k,l)*wrepp
                v(ir,43) = v(ir,43) + d_d(14,k,l)*wrepp
                v(ir,44) = v(ir,44) + d_d(15,k,l)*wrepp
!
              end select
            end do
          end do
          where (v(ir,1:limkl) /= 0.00D00)
            logv(ir,1:limkl) = .TRUE.
          end where
        end do
      end do
!
      return
      end subroutine tx_spd
