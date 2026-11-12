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

subroutine density_for_MOZYME (p, mode, nclose_loc, partp)
   !***********************************************************************
   !
   !   DENSIT COMPUTES THE DENSITY MATRIX GIVEN THE EIGENVECTOR MATRIX, AND
   !          INFORMATION ABOUT THE M.O. OCCUPANCY.
   !
   !  INPUT:  COCC  = COMPRESSED OCCUPIED EIGENVECTOR MATRIX
   !
   !   ON EXIT: P   = DENSITY MATRIX
   !
   !***********************************************************************
    use molkst_C, only: numat, mpack, keywrd
    use MOZYME_C, only : lijbo, nijbo, ncf, ncocc, &
      nncf, iorbs, cocc, icocc
    use chanel_C, only: iw
    use acc_mozyme_C, only : pd_ok, acc_path_note
    implicit none
    integer, intent (in) :: mode, nclose_loc
    double precision, dimension (mpack), intent (in) :: partp
    double precision, dimension (mpack), intent (inout) :: p
    logical :: first = .true.
    logical, save :: prnt
    integer :: i, j, j1, ja, jj, k, k1, k2, ka, kk, l, loop, nj
    double precision :: spinfa, sum
    integer, external :: ijbo
    integer, external :: acc_density_mode
    integer :: gmode
    integer(8) :: gc0, gc1, grate
    double precision :: tg
    double precision, allocatable :: pg(:)
    logical :: gok
    if (first) then
      first = .false.
      prnt = Index (keywrd, " DIAG ") /= 0
    end if
    !
    !  GPU (OpenACC) version, MOPAC_ACC_DENS set, full density only (acc_density_mozyme.F90).
    !  "test": the GPU result goes into a copy and is compared with the CPU result.
    !  The device copy of P is no longer current once P is rebuilt here on the CPU.
    !
    pd_ok = .false.
    gmode = 0
    if (mode == 0 .and. lijbo) then
      gmode = acc_density_mode()
    else if (acc_density_mode() /= 0) then
      if (.not. lijbo) then
        call acc_path_note('density on CPU (no atom-pair index table)')
      else
        call acc_path_note('density on CPU (partial updates: only some atoms move)')
      end if
    end if
    if (gmode == 1) then
      call acc_density_core (p, nclose_loc, .true., gok)
      if (gok) go to 900
      gmode = 0                                ! not enabled or no device memory: CPU below
    else if (gmode == 2) then
      allocate (pg(mpack))
      call system_clock (gc0, grate)
      call acc_density_core (pg, nclose_loc, .false., gok)
      call system_clock (gc1)
      tg = dble(gc1 - gc0)/dble(grate)
      if (.not. gok) then
        deallocate (pg)
        gmode = 0
      end if
    end if
    if (mode == 0) then
      !
      !  INITIALIZE P:  FULL DENSITY CALCULATION
      !
      p(:) = 0.d0
    else if (mode ==-1) then
      !
      !   FULL DENSITY MATRIX IS IN PARTP, REMOVE DENSITY DUE TO
      !   OLD LMO's USED IN SCF
      !
      p(:) = -0.5d0 * partp(:)
    else
      !
      !   PARTIAL DENSITY MATRIX IS IN PARTP, BUILD THE REST OF P
      !
      p(:) = 0.5d0 * partp(:)
    end if

    do i = 1, nclose_loc
      loop = ncocc(i)
      ja = 0
      if (lijbo) then
        do jj = nncf(i) + 1, nncf(i) + ncf(i)
          j = icocc(jj)
          nj = iorbs(j)
          ka = loop
          do kk = nncf(i) + 1, nncf(i) + ncf(i)
            k = icocc(kk)
            if (j == k) then
              l = nijbo (j, k)
              do j1 = 1, nj
                sum = cocc(ja+j1+loop)
                do k1 = 1, j1
                  k2 = ka + k1
                  l = l + 1
                  p(l) = p(l) + cocc(k2) * sum
                end do
              end do
            else if (j > k .and. nijbo (j, k) >= 0) then
              l = nijbo (j, k)
              do j1 = 1, nj
                sum = cocc(ja+j1+loop)
                do k1 = 1, iorbs(k)
                  k2 = ka + k1
                  l = l + 1
                  p(l) = p(l) + cocc(k2) * sum
                end do
              end do
            end if
            ka = ka + iorbs(k)
          end do
          ja = ja + nj
        end do
      else
        do jj = nncf(i) + 1, nncf(i) + ncf(i)
          j = icocc(jj)
          nj = iorbs(j)
          ka = loop
          do kk = nncf(i) + 1, nncf(i) + ncf(i)
            k = icocc(kk)
            l = ijbo (j, k)
            if (j == k) then
              do j1 = 1, nj
                sum = cocc(ja+j1+loop)
                do k1 = 1, j1
                  k2 = ka + k1
                  l = l + 1
                  p(l) = p(l) + cocc(k2) * sum
                end do
              end do
            else if (j > k .and. l >= 0) then
              do j1 = 1, nj
                sum = cocc(ja+j1+loop)
                do k1 = 1, iorbs(k)
                  k2 = ka + k1
                  l = l + 1
                  p(l) = p(l) + cocc(k2) * sum
                end do
              end do
            end if
            ka = ka + iorbs(k)
          end do
          ja = ja + nj
        end do
      end if
    end do
    if (mode == 0 .or. mode == 1) then
      !
      !    FULL DENSITY CALCULATION.  MULTIPLY BY 2 FOR SPIN
      !
      spinfa = 2.d0
    else if (mode ==-1) then
      !
      !   MAKING PARTIAL DENSITY MATRIX. REVERSE SIGN ONCE MORE
      !
      spinfa = -2.d0
    else
      spinfa = 1.d0
    end if
   !
    if (Abs(spinfa - 1.d0) > 1.d-10) then
      p(:) = spinfa * p(:)
    end if
    if (gmode == 2) then
      call acc_density_compare (p, pg, tg)
      deallocate (pg)
    end if
900 continue
    if (prnt) then
      sum = 0.d0
      if (lijbo) then
        do i = 1, numat
          j = nijbo(i, i) + 1
          if (iorbs(i) > 0) then
            sum = sum + p(j)
          end if
          if (iorbs(i) == 4) then
            sum = sum + p(j+2) + p(j+5) + p(j+9)
          end if
        end do
      else
        do i = 1, numat
          j = ijbo (i, i) + 1
          if (iorbs(i) > 0) then
            sum = sum + p(j)
          end if
          if (iorbs(i) == 4) then
            sum = sum + p(j+2) + p(j+5) + p(j+9)
          end if
        end do
      end if
      write (iw,*) " COMPUTED NUMBER OF ELECTRONS:", sum
    end if
end subroutine density_for_MOZYME
