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

subroutine hcore_for_MOZYME ()
  use molkst_C, only: numat, norbs, id, numcal, n2elec, moperr, &
     & enuclr, l1u, l2u, l3u, keywrd, efield, mpack, cutofp, line
  use cosmo_C, only : useps,  phinet, qscnet, qdenet
  use linear_cosmo, only : addnucz
  use chanel_C, only: iw
  use funcon_C, only: a0, ev, fpc_9
  use overlaps_C, only : cutof1, cutof2
  use parameters_C, only: tore, dd, natorb
  use common_arrays_C, only: h, coord, nat, w, wj => w, wk, uspd, tvec
  use MOZYME_C, only : semidr, direct, cutofs, parth, &
    iorbs, jopt, mode, numred, refnuc
!$ use omp_lib, only : omp_get_max_threads
  implicit none
!
  character (len=248) :: tmpkey
  logical :: calci, calcij, calcj, fldon, lmolaris_qmmm, exists
  logical, save :: debug
  integer :: i1, i2, ii, im1, io1, ione, ired, j, j1, jj, jo1, jred, k, &
 & krmax, kro, ks, mm, nj, kr, i, ni, itemp, imol=0, q_at, link_at, mol_at
  double precision :: const, enuc, fldcon, fnuc, half, hterme, xf, yf, zf, xj(3)
  double precision, parameter :: eps = 1.d-10
  double precision, dimension (45) :: e1b, e2a
  double precision, dimension (2025) :: wjd, wkd
  double precision, dimension (9, 9) :: di, dibits
  double precision, allocatable :: vqc(:)
  double precision, external :: reada
  integer, external :: ijbo
  integer :: nthr
  fldcon = 0.d0
  fnuc = 0.d0
  debug = (Index (keywrd, " HCORE") /= 0)
  lmolaris_qmmm = (index(keywrd,'QMMM') /= 0)
  if (lmolaris_qmmm) then
  !
  !  Read in the energy in kcal/mol that an electron on each atom would have, arising from
  !  the partial charges on all atoms.
  !
    line="mol.in"
    call add_path(line)
    inquire (file=trim(line), exist = exists)
    if (.not. exists) then
        call mopend("A file named '"//trim(line)//"' was expected, but was not found.")
        return
    end if
    open(85, file=line, form='formatted')
    read(85, *, iostat = i)
    if (i /= 0) then
        call mopend("A file named '"//trim(line)//"' exists, but is empty.")
        return
    end if
    read(85, *, iostat = i)q_at, link_at
    if (i /= 0) then
        call mopend("A file named '"//trim(line)//"' exists, but is corrupt.")
        return
    end if
    mol_at = q_at + link_at
    if(mol_at /= numat) then
      write(line,'(a,i5,a,i5,a)') &
        " The number of atoms in 'mol.in':",mol_at," and in the MOPAC data set:",numat," are different."
      call mopend(trim(line))
      write(iw,'(10x,a)')" Correct fault and resubmit."
      return
    end if
    if (allocated(vqc)) deallocate(vqc)
    allocate(vqc(numat))
    do i = 1, numat
      read(85,*)line,line,line,line,vqc(i)
      if (debug) write(iw,'(a, i4, a, f9.3)')"ATOM No.",i," VQC(I)",vqc(i)
    end do
    close(85)
  end if
  call add_more_interactions()
  if (moperr) return
  if (imol /= numcal) then
    if (index(keywrd, " SILENT") == 0 .and. (debug .or. id /= 0)) then
      write (iw, "(A,31X,F12.6,A)") " Overlap Cutoff Distance:", Sqrt(cutofs), " CUTOFS"
      write (iw, "(A,F12.6,A)") &
       & " Cutoff for quadrupolar and higher 2-electron integrals:", Sqrt(cutof2), " CUTOF2"
      write (iw, "(A,26X,F12.6,A)") " Cutoff for dipolar integrals:", Sqrt(cutof1), " CUTOF1"
      if (id /= 0) &
          write (iw, "(A,39X,F12.6,A)") " Madelung cutoff:", cutofp, " CUTOFP"
    end if
    imol = numcal
    xf = 0.d0
    yf = 0.d0
    zf = 0.d0
    tmpkey = trim(keywrd)
    i = Index (tmpkey, " FIELD(") + Index (tmpkey, " FIELD=(")
    if (i /= 0) then
       !
       !   ERASE ALL TEXT FROM TMPKEY EXCEPT FIELD DATA
       !
      tmpkey (:i) = " "
      itemp = Index (tmpkey, ")")
      tmpkey (itemp:) = " "
       !
       !   READ IN THE EFFECTIVE FIELD IN X,Y,Z COORDINATES
       !
      xf = reada (tmpkey, i)
      i = Index (tmpkey, ",")
      if (i /= 0) then
        tmpkey (i:i) = " "
        yf = reada (tmpkey, i)
        i = Index (tmpkey, ",")
        if (i /= 0) then
          tmpkey (i:i) = " "
          zf = reada (tmpkey, i)
        end if
      end if
      write (iw, "(/10X,'THE ELECTRIC FIELD IS',3F10.5, ' VOLTS/ANGSTROM',&
       &/)") xf, yf, zf
    end if
    !
    !        CONST = Ao/(8h)  (h=Hartree = eV/atomic unit)
    !
    const = a0 / ev
    !
    efield(1) = xf * const
    efield(2) = yf * const
    efield(3) = zf * const
  end if
  ione = 1
  if (id /= 0) ione = 0
  if (mode == -1) then
    h(:mpack) = -h(:mpack)
    enuclr = -enuclr
  else if (mode == 0) then
    enuclr = 0.d0
  else
    h = parth
    enuclr = refnuc
  end if
  krmax = n2elec + 101
  fldon = .false.
  if (Abs (efield(1)) > eps .or. Abs (efield(2)) > eps .or. Abs (efield(3)) > eps) then
    fldcon = eV/a0 ! = 51.42
    fldon = .true.
  end if
 !
  kr = 1
 !
  mm = 0
  ired = 1
  if (mode == 0) then
    !
    !   ZERO OUT THE H MATRIX
    !
    h(1:mpack) = 0.d0
  end if
  nthr = 1
!$ nthr = omp_get_max_threads()
  if (mode == 0 .and. id == 0 .and. semidr .and. .not. fldon .and. .not. lmolaris_qmmm .and. .not. debug &
    .and. nthr > 1 .and. numat > 1) then
    call hcore_rows_omp()           ! the loop below, over OpenMP threads, bitwise the same result
    if (moperr) return
  else
  do i = 1, numat

    calci = (jopt(ired) == i)
    if (calci .and. ired < numred) then
      ired = ired + 1
    end if
    ni = nat(i)
    if (mode == 0) then
        !
        ! FILL THE DIAGONALS, AND OFF-DIAGONALS ON THE SAME ATOM
        !
      i2 = ijbo (i, i)
      do i1 = 1, iorbs(i)
        do j1 = 1, i1
          i2 = i2 + 1
          h(i2) = 0.d0
          if (fldon) then
            io1 = i1 - 1
            jo1 = j1 - 1
            if ((jo1 == 0) .and. (io1 == 1)) then
              hterme = -a0 * dd(ni) * efield(1) * fldcon
              h(i2) = hterme
            end if
            if ((jo1 == 0) .and. (io1 == 2)) then
              hterme = -a0 * dd(ni) * efield(2) * fldcon
              h(i2) = hterme
            end if
            if ((jo1 == 0) .and. (io1 == 3)) then
              hterme = -a0 * dd(ni) * efield(3) * fldcon
              h(i2) = hterme
            end if
          end if
        end do
        mm = mm + 1
        h(i2) = uspd(mm)
        if (fldon) then
          fnuc = -(efield(1)*coord(1, i) + efield(2)*coord(2, i) + &
               & efield(3)*coord(3, i)) * fldcon
          h(i2) = h(i2) + fnuc
        else
          if(lmolaris_qmmm) h(i2) = h(i2) - vqc(i)/fpc_9
        end if
      end do
    end if
    if (fldon) then
      enuclr = enuclr - fnuc * tore(nat(i))
    end if
    if (lmolaris_qmmm) enuclr = enuclr + vqc(i)/fpc_9*tore(nat(i))
    !
    !   FILL THE ATOM-OTHER ATOM ONE-ELECTRON MATRIX<PSI(LAMBDA)|PSI(SIGMA)>
    !
    jred = 1
    im1 = i - ione
    do j = 1, im1
      half = 1.d0
      if (i == 46 .and. j < 888) then
    enuclr = enuclr
   end if
      if (i == j) half = 0.5d0
      calcj = (jopt(jred) == j)
      if (calcj .and. jred < numred) jred = jred + 1
      calcij = (calci .or. calcj .or. mode == 0)
      nj = nat(j)
      if (id == 0) then
        !
        !   Molecular system
        !
        if (ijbo(i, j) >= 0) then
          if (calcij) then
            call h1elec (ni, nj, coord(1, i), coord(1, j), di)
            ii = ijbo (i, j)
            if (i == j) then
              do i1 = 1, iorbs(i)
                do j1 = 1, i1
                  ii = ii + 1
                  h(ii) = h(ii) + di(i1, j1)
                end do
              end do
            else
              do i1 = 1, iorbs(i)
                do j1 = 1, iorbs(j)
                  ii = ii + 1
                  h(ii) = h(ii) + di(i1, j1)
                end do
              end do
            end if
            !
            ! CALCULATE THE TWO-ELECTRON INTEGRALS, W; THE ELECTRON
            ! NUCLEAR TERMS E1B AND E2A; AND THE NUCLEAR-NUCLEAR TERM ENUC.
            !
            call rotate (ni, nj, coord(1, i), coord(1, j), w(kr), i1, e1b, e2a, enuc)
            enuclr = enuclr + enuc
           !
          else if ( .not. direct) then
            kr = kr + (natorb(ni)*(natorb(ni)+1))/2 * (natorb(nj)*(natorb(nj)+1))/2
          end if
        else if (ijbo(i, j) ==-2) then
          if (calcij) then
            call outer2 (ni, nj, coord(1, i), coord(1, j), w(kr), kr, e1b, e2a, enuc, id, semidr)
            enuclr = enuclr + enuc
          else if ( .not. semidr) then
            if (natorb(ni)*natorb(nj) > 0) then
              if (natorb(ni) > 1) then
                if (natorb(nj) > 1) then
                  kr = kr + 7
                else
                  kr = kr + 4
                end if
              else if (natorb(nj) > 1) then
                kr = kr + 4
              else
                kr = kr + 1
              end if
            end if
          end if
        else if (calcij) then
          call outer1 (ni, nj, coord(1, i), coord(1, j), w(kr), kr, e1b, e2a, enuc, 0, semidr)
          enuclr = enuclr + enuc
        else if ( .not. semidr) then
          if (natorb(ni)*natorb(nj) > 0) kr = kr + 1
        end if
        if (calcij) then
          !
          !   ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM I.
          !
          ii = ijbo (i, i)
          j1 = (iorbs(i)*(iorbs(i)+1)) / 2
          do i1 = 1, j1
            ii = ii + 1
            h(ii) = h(ii) + e1b(i1) * half
          end do
          !
          !   ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM J.
          !
          ii = ijbo (j, j)
          j1 = (iorbs(j)*(iorbs(j)+1)) / 2
          do i1 = 1, j1
            ii = ii + 1
            h(ii) = h(ii) + e2a(i1) * half
          end do
        end if
        if (kr > krmax-100 .and. i /= numat) then
          write (iw,*) kr, krmax
          write (iw, "(' Running out of storage for W in HCORE ')")
          write (iw,*) " NUMBER OF ATOMS CALCULATED FOR 'W':", i
          write (iw,*) " NUMBER OF ATOMS IN SYSTEM:", numat
          call mopend ("Running out of storage for W in HCORE")
          return
        end if
      else
        !
        !   Solid-state system
        !
        if (ijbo(i, j) >= 0) then
          if (calcij) then
            di = 0.D0
            do ii = -l1u, l1u
              do jj = -l2u, l2u
                do k = -l3u, l3u
                  xj = coord(:,j) + tvec(:,1)*ii + tvec(:,2)*jj + tvec(:,3)*k
                  call h1elec (ni, nj, coord(1,i), xj, dibits)
                  di = di + dibits
                end do
              end do
            end do
            ii = ijbo (i, j)
            if (i == j) then
              do i1 = 1, iorbs(i)
                do j1 = 1, i1
                  ii = ii + 1
                  h(ii) = h(ii) + di(i1, j1)
                end do
              end do
            else
              do i1 = 1, iorbs(i)
                do j1 = 1, iorbs(j)
                  ii = ii + 1
                  h(ii) = h(ii) + di(i1, j1)
                end do
              end do
            end if
            !
            ! CALCULATE THE TWO-ELECTRON INTEGRALS, W;
            ! THE ELECTRON NUCLEAR TERMS E1B AND E2A;
            ! AND THE NUCLEAR-NUCLEAR TERM ENUC.
            !
            kro = kr
            call solrot (ni, nj, coord(1,i), coord(1,j), wjd, wkd, kr, e1b, e2a, enuc)
            w(kro:kr - 1) = wjd(:kr-kro)
            wk(kro:kr - 1) = wkd(:kr-kro)
            enuclr = enuclr + enuc
          else if (natorb(ni) == 1) then
            if (natorb(nj) == 1) then
              kr = kr + 1
            else
              kr = kr + 10
            end if
          else if (natorb(nj) == 1) then
            kr = kr + 10
          else
            kr = kr + 100
          end if
        else if (ijbo(i, j) ==-2) then
          if (calcij) then
            ks = kr
            call outer2 (ni, nj, coord(1, i), coord(1, j), wj(kr), kr, &
                 & e1b, e2a, enuc, id, semidr)
            do k = ks, kr - 1
              wk(k) = 0.d0
            end do
            enuclr = enuclr + enuc
          else if (natorb(ni)*natorb(nj) > 0) then
            if (natorb(ni) > 1) then
              if (natorb(nj) > 1) then
                kr = kr + 7
              else
                kr = kr + 4
              end if
            else if (natorb(nj) > 1) then
              kr = kr + 4
            else
              kr = kr + 1
            end if
          end if
        else if (calcij) then
          wk(kr) = 0.d0
          call outer1 (ni, nj, coord(1, i), coord(1, j), wj(kr), kr, &
               & e1b, e2a, enuc, id, semidr)
          enuclr = enuclr + enuc
        else if (natorb(ni)*natorb(nj) > 0) then
          kr = kr + 1
        end if
        if (calcij) then
           !
           !   ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM I.
           !
          ii = ijbo (i, i)
          j1 = (iorbs(i)*(iorbs(i)+1)) / 2
          do i1 = 1, j1
            ii = ii + 1
            h(ii) = h(ii) + e1b(i1) * half
          end do
           !
           !   ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM J.
           !
          ii = ijbo (j, j)
          j1 = (iorbs(j)*(iorbs(j)+1)) / 2
          do i1 = 1, j1
            ii = ii + 1
            h(ii) = h(ii) + e2a(i1) * half
          end do
        end if
        if (kr > krmax-100 .and. i /= numat) then
          write (iw,*) kr, krmax
          write (iw, "(' Running out of storage for W in HCORE ')")
          write (iw,*) " NUMBER OF ATOMS CALCULATED FOR 'W':", i
          write (iw,*) " NUMBER OF ATOMS IN SYSTEM:", numat
          call mopend ("Running out of storage for W in HCORE")
        end if
      end if
    end do
    ii = iorbs(i)
    ii = (ii*(ii+1)) / 2
    if (id /= 0) then
      do i1 = kr, kr + ii * ii - 1
        wk(i1) = 0.d0
      end do
    end if
    if (ii /= 0) then
      call wstore (w(kr), kr, ni, ii)
    end if
  end do
  end if
 !
 !
  if (mode == -1) then
    parth(:mpack) = -h(:mpack)
    refnuc = -enuclr
  end if
  if (useps) then
    ! In the following routine the dielectric correction to the core-core-
    ! interaction is added to ENUCLR (just set arrays to zero)

    call addnucz (phinet, qscnet, qdenet)

  end if
  kr = kr - 1
 !
 !
  if (debug) then
    write (iw, "(//10X,'ONE-ELECTRON MATRIX FROM HCORE')")
    if (mode ==-1) then
      write (iw, "(10X,A)") " AFTER REMOVAL OF TERMS FOR MOVING ATOMS"
    end if
    if (mode == 1) then
      write (iw, "(10X,A)") " AFTER ADDITION OF TERMS FOR MOVING ATOMS"
    end if
    call vecprt_for_MOZYME (h, norbs)
    if (kr > 2000) then
      write (iw,*) " THE TWO-ELECTRON MATRIX IS TOO LARGE TO PRINT"
    end if
    j = Min (kr, 2000)
    if (id == 0) then
      write (iw, "(//10X,'TWO-ELECTRON MATRIX IN HCORE'/)")
10000   format (10 f8.4)
      write (iw, 10000) (w(i), i=1, j)
    else
      write (iw, "(//10X,'TWO-ELECTRON J MATRIX IN HCORE'/)")
      write (iw, 10000) (wj(i), i=1, j)
      write (iw, "(//10X,'TWO-ELECTRON K MATRIX IN HCORE'/)")
      write (iw, 10000) (wk(i), i=1, j)
    end if
  end if
contains
  subroutine hcore_rows_omp()
! OpenMP version of the atom-pair loop of hcore_for_MOZYME, for a molecular system (id = 0) in the
! default direct mode (semidr: rotate, outer1 and outer2 use w only as scratch and do not advance kr,
! which changes only in wstore), mode 0, without an electric field, a QMMM potential or HCORE output.
! Rows i are taken in blocks. Within a block the pairs (i, j < i) are computed in parallel, with
! private scratch: each off-diagonal block of h belongs to one pair and is written directly; the
! diagonal block of row i (one-center term, then e1b for j = 1, ..., i-1) is summed per row; the
! e2a terms and the core-core energies are kept, and added afterwards in the serial order (rows in
! order: enuclr, then the diagonal blocks of the atoms j < i). h, enuclr and the stored one-center
! integrals (wstore) are therefore bitwise those of the serial loop for any number of threads. (Only
! the scratch left in w beyond the last stored integrals differs; nothing reads it.)
    integer :: i, j, nb, i0, nrow, r, t, k, ii, i1, j1, idum, ni_, nj_
    integer(8) :: per_row, base
    integer, allocatable :: tri(:), ptri(:), morb(:)
    double precision, allocatable :: dsum(:,:), e2buf(:), enb(:,:)
    double precision :: e1l(45), e2l(45), enl, wloc(2025), dil(9,9), xs(3), half1, x
    double precision, external :: ss
    half1 = 1.d0
    allocate (tri(numat), ptri(numat + 1), morb(numat + 1))
    ptri(1) = 0
    morb(1) = 0
    do i = 1, numat
      tri(i) = (iorbs(i)*(iorbs(i) + 1))/2
      ptri(i + 1) = ptri(i) + tri(i)
      morb(i + 1) = morb(i) + iorbs(i)
    end do
! Rows per block: about 64 MB of buffers (e2a terms and core-core energies per row), at least 4 per thread.
    per_row = int(ptri(numat + 1), 8) + int(numat, 8)
    nb = int(min(int(numat, 8), max(int(4*nthr, 8), 8000000_8/max(1_8, per_row))))
    allocate (dsum(45, nb), e2buf(int(nb, 8)*int(ptri(numat + 1), 8) + 1), enb(numat, nb))
! One-time initializations inside the integral routines (saved state) happen here, before any thread
! runs: diat sets its element table once per calculation, ss its binomials on its first call
! (diat and ss run inside the critical section below, but this keeps the order of initialization fixed).
    xs = (/1.d0, 0.d0, 0.d0/)
    call diat (nat(1), nat(1), xs, dil)
    x = ss (1, 1, 1, 1, 1, 1.d0, 1.d0, 1.d0, a0)
    do i0 = 1, numat, nb
      nrow = min(nb, numat - i0 + 1)
!$omp parallel do schedule(dynamic, 1) default(shared) &
!$omp private(r, i, j, k, ii, i1, j1, t, idum, ni_, nj_, base, e1l, e2l, enl, wloc, dil)
      do r = 1, nrow
        i = i0 + r - 1
        ni_ = nat(i)
        dsum(1:max(1, tri(i)), r) = 0.d0
        do i1 = 1, iorbs(i)
          dsum((i1*(i1 + 1))/2, r) = uspd(morb(i) + i1)
        end do
        base = int(r - 1, 8)*int(ptri(numat + 1), 8)
        do j = 1, i - 1
          nj_ = nat(j)
          k = ijbo(i, j)
          if (k >= 0) then
! h1elec -> diat2 -> set use module variables of overlaps_C (isp, ips, a, b, sa, sb) as scratch, so
! overlaps are computed one at a time (pairs within the overlap cutoff only). Making those variables
! threadprivate instead changes nvfortran's code for them (FMA contraction) and so the last bits of h.
!$omp critical (hcore_overlap)
            call h1elec (ni_, nj_, coord(1, i), coord(1, j), dil)
!$omp end critical (hcore_overlap)
            ii = k
            do i1 = 1, iorbs(i)
              do j1 = 1, iorbs(j)
                ii = ii + 1
                h(ii) = h(ii) + dil(i1, j1)
              end do
            end do
            idum = 1
            call rotate (ni_, nj_, coord(1, i), coord(1, j), wloc, idum, e1l, e2l, enl)
          else if (k == -2) then
            idum = 1
            call outer2 (ni_, nj_, coord(1, i), coord(1, j), wloc, idum, e1l, e2l, enl, id, semidr)
          else
            idum = 1
            call outer1 (ni_, nj_, coord(1, i), coord(1, j), wloc, idum, e1l, e2l, enl, 0, semidr)
          end if
          enb(j, r) = enl
          do t = 1, tri(i)
            dsum(t, r) = dsum(t, r) + e1l(t)*half1
          end do
          do t = 1, tri(j)
            e2buf(base + ptri(j) + t) = e2l(t)
          end do
        end do
      end do
!$omp end parallel do
      do r = 1, nrow
        i = i0 + r - 1
        if (tri(i) > 0) then
          ii = ijbo(i, i)
          do t = 1, tri(i)
            h(ii + t) = dsum(t, r)
          end do
        end if
        base = int(r - 1, 8)*int(ptri(numat + 1), 8)
! enuclr is summed in this loop, not in a loop of its own: a bare sum loop may be vectorized with
! reassociated additions (nvfortran without -Kieee), which changes the last bits.
        do j = 1, i - 1
          enuclr = enuclr + enb(j, r)
          ii = ijbo(j, j)
          do t = 1, tri(j)
            h(ii + t) = h(ii + t) + e2buf(base + ptri(j) + t)*half1
          end do
        end do
        if (i > 1 .and. kr > krmax - 100 .and. i /= numat) then
          write (iw,*) kr, krmax
          write (iw, "(' Running out of storage for W in HCORE ')")
          write (iw,*) " NUMBER OF ATOMS CALCULATED FOR 'W':", i
          write (iw,*) " NUMBER OF ATOMS IN SYSTEM:", numat
          call mopend ("Running out of storage for W in HCORE")
          deallocate (tri, ptri, morb, dsum, e2buf, enb)
          return
        end if
        if (tri(i) /= 0) call wstore (w(kr), kr, nat(i), tri(i))
      end do
    end do
    deallocate (tri, ptri, morb, dsum, e2buf, enb)
  end subroutine hcore_rows_omp
end subroutine hcore_for_MOZYME
