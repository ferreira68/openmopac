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

! OpenACC (GPU) version of the computational part of diagg1 (lijbo case): virtual LMO energies,
! occupied-virtual LMO interaction matrix elements, and (when requested) occupied LMO energies.
!
! The control logic and the saved state (cutoff, flim, oldlim, safety, fref) stay in diagg1, which
! passes them in. The results are those of the CPU loops:
!   eigv(1:nvir), eigs_occ(1:nocc) (if docc), and, for a full pass, the list ifmo/fmo in (i, j)
!   order with the nij cap, nfmo(1:nvir), ijc, sumt and tiny; for a reuse pass, fmo(1:ijc).
! Each matrix element and energy is accumulated in the same order as on the CPU, except where a
! sum runs over the atoms of an LMO in parallel (eigv, eigs_occ, sumt).
!
! Device work:
!   K0  per occupied LMO: aocc (squared atomic contributions) and atom offsets within the LMO.
!   K1  per virtual LMO (gang), atoms in parallel: w = F c over the LMO's own atoms (compact,
!       same layout as cvir), squared atomic contributions avir and aov, and eigv.
!   K2  full pass: one thread per (virtual, occupied) pair: the screening test of diagg1 and, for
!       pairs that pass, the matrix element; survivors are appended with an atomic counter.
!       The host sorts them into (i, j) order and applies the nij cap exactly as the CPU loop does.
!   K3  reuse pass: the matrix elements of the existing list ifmo, one thread each.
!   K4  occupied LMO energies, one gang per LMO.
! If the candidate buffer overflows, ok = .false. is returned and diagg1 runs on the CPU.
!
! Enabled by the environment variable MOPAC_ACC_DIAGG ("test": run both and compare).

integer function acc_diagg_mode()
  implicit none
  integer, save :: mode = -1
#ifdef _OPENACC
  character(len=16) :: env
  integer :: stat
  if (mode < 0) then
    call get_environment_variable("MOPAC_ACC_DIAGG", env, status=stat)
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
  acc_diagg_mode = mode
end function acc_diagg_mode

subroutine acc_diagg1_core(fao, nocc, nvir, eigv, eigs_occ, nfmo_out, ifmo, ifmo_in, fmo, fmo_dim, nij, &
  cutoff, flim, oldlim, full, docc, ijc, sumt, tiny, ok)
#ifdef _OPENACC
  use molkst_C, only : numat, mpack
  use MOZYME_C, only : ncf, nce, nncf, nnce, ncocc, ncvir, icocc, icvir, cocc, cvir, iorbs, nfmo
  use common_arrays_C, only : nfirst, nlast, p
  use acc_mozyme_C, only : nij_d => nij, acc_mozyme_sync, acc_mozyme_buffers, acc_to_device, acc_matches_device, &
    fd_c, pd_c, fd_ok, pd_ok, allow_diagg, held_cand, eimp_dev
  use common_arrays_C, only : fcom => f
  use iso_c_binding, only : c_f_pointer
  use chanel_C, only : iw
#endif
  implicit none
  integer, intent(in) :: nocc, nvir, fmo_dim, nij
  double precision, intent(in) :: fao(*), cutoff, flim, oldlim
  logical, intent(in) :: full, docc
  double precision, intent(out) :: eigv(*), eigs_occ(*)
  integer, intent(inout) :: nfmo_out(*)        ! written on a full pass only
  integer, intent(out) :: ifmo(2, fmo_dim)        ! written on a full pass
  integer, intent(in) :: ifmo_in(2, fmo_dim)      ! read on a reuse pass (may be the same array as ifmo)
  double precision, intent(out) :: fmo(fmo_dim)
  integer, intent(out) :: ijc
  double precision, intent(out) :: sumt, tiny
  logical, intent(out) :: ok
#ifdef _OPENACC
  logical :: dummy
  integer :: cap, ncand
  integer, allocatable, save :: ci(:), cj(:), cl(:)
  double precision, allocatable, save :: cv(:)
  integer, save :: capsave = 0
  integer :: nlist, attempt
  double precision, pointer, contiguous :: fdev(:), pdev(:)
  logical :: fstale, pstale
  integer, save :: nstale = 0, nreuse = 0
  ok = .false.
  dummy = acc_mozyme_sync()
  if (.not. allow_diagg) then           ! not enabled, or not enough device memory: CPU
    if (allocated(ci)) then
      !$acc exit data delete(ci, cj, cl, cv)
      deallocate (ci, cj, cl, cv)
      capsave = 0
      held_cand = 0
    end if
    call acc_eimp_host_fallback()       ! the CPU diagg1 reads the eimp values from the host P
    return
  end if
  ok = .true.
! F and P on the device: reuse the buffers left by the GPU Fock build if they hold the current
! common_arrays f and p (checked by flag, address and a sampled comparison); otherwise copy.
  if (fd_ok .and. pd_ok .and. allocated(fcom)) then
    if (loc(fao(1)) /= loc(fcom(1))) then
      fd_ok = .false.
    else
! eimp replaces the first element of each off-diagonal block of P by the sum of squares of that
! block of F, which diagg1 uses for screening. The device copy still holds the density of the Fock
! build. If the host eimp ran, apply it here too and then compare; if it was skipped (eimp_dev,
! acc_eimp_skip), compare first (both copies hold the density), then apply it on the device only.
      call c_f_pointer(fd_c, fdev, [mpack])
      call c_f_pointer(pd_c, pdev, [mpack])
      if (.not. eimp_dev) call acc_eimp_kernel(numat, mpack, fdev, pdev, nij_d, iorbs)
      fstale = .not. acc_matches_device(fd_c, fao, mpack, 0.d0)
      pstale = .not. acc_matches_device(pd_c, p, mpack, 1.d-12)
      if (eimp_dev .and. .not. (fstale .or. pstale)) call acc_eimp_kernel(numat, mpack, fdev, pdev, nij_d, iorbs)
      if (fstale .or. pstale) then
        nstale = nstale + 1
        if (nstale <= 20) write (iw, '(a,i0,a,l2,a,l2)') ' ACC DIAGG: device copy stale (occurrence ', nstale, &
          '): F', fstale, '  P', pstale
        fd_ok = .false.
      end if
    end if
  else
    fd_ok = .false.
  end if
  if (.not. fd_ok) then
    if (.not. acc_mozyme_buffers(mpack)) then
      ok = .false.
      call acc_eimp_host_fallback()
      return
    end if
    call acc_to_device(fd_c, fao, mpack)
    call acc_to_device(pd_c, p, mpack)
    if (eimp_dev) then                  ! the host P lacks the eimp values: apply them on the device
      call c_f_pointer(fd_c, fdev, [mpack])
      call c_f_pointer(pd_c, pdev, [mpack])
      call acc_eimp_kernel(numat, mpack, fdev, pdev, nij_d, iorbs)
    end if
  else
    nreuse = nreuse + 1
    if (nreuse == 1) write (iw, '(a)') ' ACC DIAGG: using the device copies of F and P from the GPU Fock build'
  end if
  call c_f_pointer(fd_c, fdev, [mpack])
  call c_f_pointer(pd_c, pdev, [mpack])
  cap = max(1000000, 2*fmo_dim, capsave)
  do attempt = 1, 2
  if (cap /= capsave) then
    if (allocated(ci)) then
      !$acc exit data delete(ci, cj, cl, cv)
      deallocate (ci, cj, cl, cv)
    end if
    allocate (ci(cap), cj(cap), cl(cap), cv(cap))
    !$acc enter data create(ci, cj, cl, cv)
    capsave = cap
    held_cand = 20_8*int(cap, 8)
  end if
  nlist = 0
  if (.not. full) nlist = min(sum(nfmo(1:nvir)), nij)
  call acc_diagg1_kernels(numat, mpack, nocc, nvir, size(icocc), size(cocc), size(icvir), size(cvir), &
    fdev, pdev, nij_d, iorbs, nfirst, nlast, ncf, nncf, ncocc, icocc, cocc, nce, nnce, ncvir, icvir, cvir, &
    cutoff, flim, full, docc, cap, ci, cj, cl, cv, ncand, eigv, eigs_occ, nlist, ifmo_in, fmo, fmo_dim, sumt, tiny)
  if (.not. full .or. ncand <= cap) exit
! More candidates than buffer space: enlarge the buffer and repeat once.
  write (iw, '(a,i0,a,i0,a)') ' ACC DIAGG: ', ncand, ' candidate pairs exceed the buffer (', cap, '); enlarging it'
  cap = ncand + ncand/4
  end do
! The density changes after diagg1 (diagg2, density_for_MOZYME): the buffers are no longer current.
  fd_ok = .false.
  pd_ok = .false.
  if (full) then
    if (ncand > cap) then
      write (iw, '(a)') ' ACC DIAGG: candidate buffer still too small; this DIAGG1 call runs on the CPU'
      ok = .false.
      call acc_eimp_host_fallback()
      return
    end if
    call acc_diagg1_order(ncand, nvir, ci, cj, cl, cv, nij, oldlim, fmo_dim, ifmo, fmo, nfmo_out, ijc, sumt, tiny)
  else
    ijc = nlist                     ! nfmo is unchanged by a reuse pass
  end if
#else
  ok = .false.
#endif
end subroutine acc_diagg1_core

#ifdef _OPENACC
subroutine acc_eimp_kernel(numat, mpack, f, p, nij, iorbs)
! Device version of eimp: p(k+1) = sum of f(m)**2 over the off-diagonal block (i, j), k = nijbo(i, j).
  implicit none
  integer, intent(in) :: numat, mpack, nij(numat, numat), iorbs(numat)
  double precision, intent(in) :: f(mpack)
  double precision, intent(inout) :: p(mpack)
  integer :: i, j, k, l, m
  double precision :: s
  !$acc parallel loop gang copyin(iorbs) present(nij) deviceptr(f, p)
  do i = 1, numat
    !$acc loop vector private(k, l, m, s)
    do j = 1, i - 1
      k = nij(i, j)
      if (k >= 0) then
        l = iorbs(i)*iorbs(j)
        if (l /= 0) then
          s = 0.d0
          do m = k + 1, k + l
            s = s + f(m)**2
          end do
          p(k + 1) = s
        end if
      end if
    end do
  end do
end subroutine acc_eimp_kernel

subroutine acc_diagg1_kernels(numat, mpack, nocc, nvir, licocc, lcocc, licvir, lcvir, &
  fao, p, nij, iorbs, nfirst, nlast, ncf, nncf, ncocc, icocc, cocc, nce, nnce, ncvir, icvir, cvir, &
  cutoff, flim, full, docc, cap, ci, cj, cl, cv, ncand, eigv, eigs_occ, nlist, ifmo_in, fmo, fmo_dim, sumt, tiny)
  implicit none
  integer, intent(in) :: numat, mpack, nocc, nvir, licocc, lcocc, licvir, lcvir, cap, nlist, fmo_dim
  double precision, intent(in) :: fao(mpack), p(mpack), cocc(lcocc), cvir(lcvir), cutoff, flim
  integer, intent(in) :: nij(numat, numat), iorbs(numat), nfirst(numat), nlast(numat)
  integer, intent(in) :: ncf(nocc), nncf(nocc), ncocc(nocc), icocc(licocc)
  integer, intent(in) :: nce(nvir), nnce(nvir), ncvir(nvir), icvir(licvir)
  logical, intent(in) :: full, docc
  integer, intent(inout) :: ci(cap), cj(cap), cl(cap)
  double precision, intent(inout) :: cv(cap)
  integer, intent(out) :: ncand
  double precision, intent(out) :: eigv(nvir), eigs_occ(nocc)
  integer, intent(in) :: ifmo_in(2, fmo_dim)
  double precision, intent(inout) :: fmo(fmo_dim)
  double precision, intent(inout) :: sumt, tiny
  double precision, allocatable :: aocc(:), avs(:), aovs(:), wsv(:)
  integer, allocatable :: offo(:), offv(:)
  integer :: cnt(1)
  integer :: i, j, jj, kk, k, k1, j1, j2, i1, i2, l, lo, n0, ne, kj, kl, lj, ii, i4, jx, j4, x, m, pos, idx, base, nk
  integer :: i1j1, i1j2, i2j1, i2j2
  double precision :: s, t, sum1
  logical :: lij
  allocate (aocc(licocc), offo(licocc), avs(licvir), aovs(licvir), offv(licvir), wsv(lcvir))
  cnt(1) = 0
  ncand = 0
  !$acc data copyin(iorbs, nfirst, nlast, ncf, nncf, ncocc, icocc, cocc, nce, nnce, ncvir, icvir, cvir) &
  !$acc& present(nij, ci, cj, cl, cv) deviceptr(fao, p) create(aocc, offo, avs, aovs, offv, wsv) copyout(eigv) copy(cnt)
!
! K0: occupied LMOs: aocc and atom offsets
!
  !$acc parallel loop gang vector private(l, s, kk, k1, k)
  do j = 1, nocc
    l = 0
    do kk = nncf(j) + 1, nncf(j) + ncf(j)
      k1 = icocc(kk)
      offo(kk) = l
      s = 0.d0
      do k = nfirst(k1), nlast(k1)
        l = l + 1
        s = s + cocc(l + ncocc(j))**2
      end do
      aocc(kk) = s
    end do
  end do
!
! K1: virtual LMOs: avir, w = F c, aov, eigv
!
  !$acc parallel loop gang vector_length(32) private(n0, ne, lo, l, s)
  do i = 1, nvir
    n0 = nnce(i)
    ne = nce(i)
    lo = ncvir(i)
    l = 0
    !$acc loop seq
    do jj = 1, ne
      j1 = icvir(n0 + jj)
      offv(n0 + jj) = l
      s = 0.d0
      do k = l + 1, l + iorbs(j1)
        s = s + cvir(k + lo)**2
      end do
      avs(n0 + jj) = s
      l = l + iorbs(j1)
    end do
    !$acc loop vector private(j1, lj, kk, k1, kj, kl, ii, i4, jx, s)
    do jj = 1, ne
      j1 = icvir(n0 + jj)
      lj = lo + offv(n0 + jj)
      do jx = 1, iorbs(j1)
        wsv(lj + jx) = 0.d0
      end do
      do kk = 1, ne
        k1 = icvir(n0 + kk)
        kj = nij(k1, j1)
        kl = lo + offv(n0 + kk)
        if (kj >= 0) then
          if (avs(n0 + kk)*p(kj + 1) > cutoff) then
            if (iorbs(k1) == 1 .and. iorbs(j1) == 1) then
              wsv(lj + 1) = wsv(lj + 1) + fao(kj + 1)*cvir(kl + 1)
            else if (k1 > j1) then
              ii = kj
              do i4 = 1, iorbs(k1)
                do jx = 1, iorbs(j1)
                  ii = ii + 1
                  wsv(lj + jx) = wsv(lj + jx) + fao(ii)*cvir(kl + i4)
                end do
              end do
            else if (k1 < j1) then
              ii = kj
              do jx = 1, iorbs(j1)
                do i4 = 1, iorbs(k1)
                  ii = ii + 1
                  wsv(lj + jx) = wsv(lj + jx) + fao(ii)*cvir(kl + i4)
                end do
              end do
            else
              do jx = 1, iorbs(j1)
                do i4 = 1, iorbs(j1)
                  if (i4 > jx) then
                    ii = kj + (i4*(i4 - 1))/2 + jx
                  else
                    ii = kj + (jx*(jx - 1))/2 + i4
                  end if
                  wsv(lj + jx) = wsv(lj + jx) + fao(ii)*cvir(kl + i4)
                end do
              end do
            end if
          end if
        end if
      end do
      s = 0.d0
      do k = nfirst(j1), nlast(j1)
        s = s + wsv(lj + k - nfirst(j1) + 1)**2
      end do
      aovs(n0 + jj) = s
    end do
    s = 0.d0
    !$acc loop vector private(k1, kl, t) reduction(+:s)
    do kk = 1, ne
      k1 = icvir(n0 + kk)
      if (aovs(n0 + kk)*avs(n0 + kk) > cutoff) then
        kl = lo + offv(n0 + kk)
        t = 0.d0
        do k = 1, nlast(k1) - nfirst(k1) + 1
          t = t + wsv(kl + k)*cvir(kl + k)
        end do
        s = s + t
      end if
    end do
    eigv(i) = s
  end do
!
! K2: full pass, occupied-virtual matrix elements (screening as in diagg1)
!
  if (full) then
    !$acc parallel loop collapse(2) gang vector private(i1, i2, j1, j2, i1j1, i1j2, i2j1, i2j2, s, lij, l, kk, k1, pos, &
    !$acc& m, nk, base, idx)
    do i = 1, nvir
      do j = 1, nocc
        i1 = icvir(nnce(i) + 1)
        if (nce(i) > 1) then
          i2 = icvir(nnce(i) + 2)
        else
          i2 = i1
        end if
        j1 = icocc(nncf(j) + 1)
        if (ncf(j) > 1) then
          j2 = icocc(nncf(j) + 2)
        else
          j2 = j1
        end if
        i1j1 = nij(i1, j1)
        i1j2 = nij(i1, j2)
        i2j1 = nij(i2, j1)
        i2j2 = nij(i2, j2)
        if (i1j1 >= 0 .or. i1j2 >= 0 .or. i2j1 >= 0 .or. i2j2 >= 0) then
          s = 0.d0
          if (i1j1 >= 0) s = abs(fao(i1j1 + 1))
          if (i1j2 >= 0) s = s + abs(fao(i1j2 + 1))
          if (i2j1 >= 0) s = s + abs(fao(i2j1 + 1))
          if (i2j2 >= 0) s = s + abs(fao(i2j2 + 1))
          if (s >= flim) then
            lij = .false.
            s = 0.d0
            l = 0
            do kk = nncf(j) + 1, nncf(j) + ncf(j)
              k1 = icocc(kk)
              pos = 0
              do m = 1, nce(i)
                if (icvir(nnce(i) + m) == k1) then
                  pos = m
                  exit
                end if
              end do
              if (pos == 0) then
                l = l + iorbs(k1)
              else if (aocc(kk)*aovs(nnce(i) + pos) < cutoff) then
                l = l + iorbs(k1)
              else
                lij = .true.
                base = ncvir(i) + offv(nnce(i) + pos)
                nk = nlast(k1) - nfirst(k1) + 1
                do m = 1, nk
                  l = l + 1
                  s = s + wsv(base + m)*cocc(l + ncocc(j))
                end do
              end if
            end do
            !$acc atomic capture
            idx = cnt(1)
            cnt(1) = cnt(1) + 1
            !$acc end atomic
            idx = idx + 1
            if (idx <= cap) then
              ci(idx) = i
              cj(idx) = j
              cv(idx) = s
              cl(idx) = 0
              if (lij) cl(idx) = 1
            end if
          end if
        end if
      end do
    end do
  end if
!
! K3: reuse pass, elements of the existing list
!
  if (.not. full .and. nlist > 0) then
    sumt = 0.d0
    tiny = 0.d0
    !$acc parallel loop gang vector copyin(ifmo_in(:, 1:nlist)) copyout(fmo(1:nlist)) private(i, j, s, l, kk, k1, pos, m, &
    !$acc& base, nk) reduction(+:sumt) reduction(max:tiny)
    do x = 1, nlist
      i = ifmo_in(1, x)
      j = ifmo_in(2, x)
      s = 0.d0
      l = 0
      do kk = nncf(j) + 1, nncf(j) + ncf(j)
        k1 = icocc(kk)
        pos = 0
        do m = 1, nce(i)
          if (icvir(nnce(i) + m) == k1) then
            pos = m
            exit
          end if
        end do
        if (pos == 0) then
          l = l + iorbs(k1)
        else if (aovs(nnce(i) + pos)*aocc(kk) < cutoff) then
          l = l + iorbs(k1)
        else
          base = ncvir(i) + offv(nnce(i) + pos)
          nk = nlast(k1) - nfirst(k1) + 1
          do m = 1, nk
            l = l + 1
            s = s + wsv(base + m)*cocc(l + ncocc(j))
          end do
        end if
      end do
      fmo(x) = s
      sumt = sumt + abs(s)
      tiny = max(tiny, abs(s))
    end do
  else if (.not. full) then
    sumt = 0.d0
    tiny = 0.d0
  end if
!
! K4: occupied LMO energies
!
  if (docc) then
    !$acc parallel loop gang vector_length(32) copyout(eigs_occ(1:nocc)) private(s)
    do i = 1, nocc
      s = 0.d0
      !$acc loop vector private(j1, l, t, kk, k1, kl, kj, jx, i4, j4, ii, sum1) reduction(+:s)
      do jj = nncf(i) + 1, nncf(i) + ncf(i)
        j1 = icocc(jj)
        l = ncocc(i) + offo(jj)
        t = 0.d0
        do kk = nncf(i) + 1, nncf(i) + ncf(i)
          k1 = icocc(kk)
          kl = ncocc(i) + offo(kk)
          kj = nij(k1, j1)
          if (kj >= 0) then
            if (aocc(kk)*p(kj + 1)*aocc(kk) >= cutoff) then
              if (k1 > j1) then
                do jx = 1, iorbs(j1)
                  sum1 = 0.d0
                  do i4 = 1, iorbs(k1)
                    ii = kj + (i4 - 1)*iorbs(j1) + jx
                    sum1 = sum1 + fao(ii)*cocc(kl + i4)
                  end do
                  t = t + cocc(l + jx)*sum1
                end do
              else if (k1 < j1) then
                do jx = 1, iorbs(j1)
                  sum1 = 0.d0
                  do i4 = 1, iorbs(k1)
                    ii = kj + (jx - 1)*iorbs(k1) + i4
                    sum1 = sum1 + fao(ii)*cocc(kl + i4)
                  end do
                  t = t + cocc(l + jx)*sum1
                end do
              else
                do jx = 1, iorbs(j1)
                  sum1 = 0.d0
                  do j4 = 1, jx
                    ii = kj + (jx*(jx - 1))/2 + j4
                    sum1 = sum1 + fao(ii)*cocc(kl + j4)
                  end do
                  do i4 = jx + 1, iorbs(k1)
                    ii = kj + (i4*(i4 - 1))/2 + jx
                    sum1 = sum1 + fao(ii)*cocc(kl + i4)
                  end do
                  t = t + cocc(l + jx)*sum1
                end do
              end if
            end if
          end if
        end do
        s = s + t
      end do
      eigs_occ(i) = s
    end do
  end if
  !$acc end data
  ncand = cnt(1)
  if (full .and. ncand > 0 .and. ncand <= cap) then
    !$acc update host(ci(1:ncand), cj(1:ncand), cl(1:ncand), cv(1:ncand))
  end if
  deallocate (aocc, offo, avs, aovs, offv, wsv)
end subroutine acc_diagg1_kernels

subroutine acc_diagg1_order(ncand, nvir, ci, cj, cl, cv, nij, oldlim, fmo_dim, ifmo, fmo, nfmo_out, ijc, sumt, tiny)
! Puts the candidates from K2 in (i, j) order and applies the logic of the CPU loop, including
! the nij cap (once ijc = nij, later pairs are neither evaluated nor counted in sumt and tiny).
  implicit none
  integer, intent(in) :: ncand, nvir, nij, fmo_dim
  integer, intent(in) :: ci(ncand), cj(ncand), cl(ncand)
  double precision, intent(in) :: cv(ncand), oldlim
  integer, intent(out) :: ifmo(2, fmo_dim), nfmo_out(nvir), ijc
  double precision, intent(out) :: fmo(fmo_dim), sumt, tiny
  integer, allocatable :: start(:), ord(:)
  integer :: i, k, m, n, nf, key
  allocate (start(nvir + 1), ord(max(1, ncand)))
! counting sort on i
  start = 0
  do k = 1, ncand
    start(ci(k) + 1) = start(ci(k) + 1) + 1
  end do
  start(1) = 1
  do i = 1, nvir
    start(i + 1) = start(i + 1) + start(i)
  end do
  do k = 1, ncand
    i = ci(k)
    ord(start(i)) = k
    start(i) = start(i) + 1
  end do
  do i = nvir, 1, -1
    start(i + 1) = start(i)
  end do
  start(1) = 1
! insertion sort on j within each i
  do i = 1, nvir
    do m = start(i) + 1, start(i + 1) - 1
      key = ord(m)
      n = m - 1
      do while (n >= start(i))
        if (cj(ord(n)) <= cj(key)) exit
        ord(n + 1) = ord(n)
        n = n - 1
      end do
      ord(n + 1) = key
    end do
  end do
  ijc = 0
  sumt = 0.d0
  tiny = 0.d0
  do i = 1, nvir
    nf = 0
    do m = start(i), start(i + 1) - 1
      if (ijc == nij) exit
      k = ord(m)
      sumt = sumt + abs(cv(k))
      tiny = max(tiny, abs(cv(k)))
      if (cl(k) == 1) then
        if (abs(cv(k)) > oldlim) then
          nf = nf + 1
          ijc = ijc + 1
          ifmo(1, ijc) = i
          ifmo(2, ijc) = cj(k)
          fmo(ijc) = cv(k)
        end if
      end if
    end do
    nfmo_out(i) = nf
  end do
  deallocate (start, ord)
end subroutine acc_diagg1_order
#endif

subroutine acc_diagg1_compare(nvir, nocc, eigv, eigvg, eigso, eigsg, docc, ijc, ijcg, ifmo, ifmog, fmo, fmog, &
  nfmo, nfmog, sumt, sumtg, tiny, tinyg, tg)
! MOPAC_ACC_DIAGG=test: CPU vs GPU results of diagg1.
  use chanel_C, only : iw
  implicit none
  integer, intent(in) :: nvir, nocc, ijc, ijcg, ifmo(2, *), ifmog(2, *), nfmo(*), nfmog(*)
  double precision, intent(in) :: eigv(*), eigvg(*), eigso(*), eigsg(*), fmo(*), fmog(*), sumt, sumtg, tiny, tinyg, tg
  logical, intent(in) :: docc
  integer :: k, nbad
  double precision :: dv, dso, df
  dv = 0.d0
  dso = 0.d0
  df = 0.d0
  nbad = 0
  do k = 1, nvir
    dv = max(dv, abs(eigv(k) - eigvg(k)))
    if (nfmo(k) /= nfmog(k)) nbad = nbad + 1
  end do
  if (docc) then
    do k = 1, nocc
      dso = max(dso, abs(eigso(k) - eigsg(k)))
    end do
  end if
  do k = 1, min(ijc, ijcg)
    if (ifmo(1, k) /= ifmog(1, k) .or. ifmo(2, k) /= ifmog(2, k)) nbad = nbad + 1
    df = max(df, abs(fmo(k) - fmog(k)))
  end do
  write (iw, '(a,es9.2,a,es9.2,a,es9.2,a,i0,a,i0,a,i0,a,es9.2,a,es9.2,a,f7.3,a)') ' ACC DIAGG TEST: max |d eigv| ', dv, &
    '  |d eigs| ', dso, '  |d fmo| ', df, '  ijc ', ijc, '/', ijcg, '  list mismatches ', nbad, '  d sumt ', abs(sumt - sumtg), &
    '  d tiny ', abs(tiny - tinyg), '  GPU ', tg, ' s'
  flush (iw)
end subroutine acc_diagg1_compare
