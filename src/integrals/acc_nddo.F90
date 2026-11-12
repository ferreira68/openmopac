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

! OpenACC prototype for the NDDO two-center integrals (rotate and its callees, marked
! "!$acc routine seq"). Compiled only with OpenACC (_OPENACC); otherwise the routines are stubs.
!
! acc_update_nddo   copies the module data read by the device routines to the device.
! acc_integral_test if the environment variable MOPAC_ACC_INTEGRAL_TEST is set, evaluates
!                   rotate for every atom pair whose atoms both lack d orbitals, on the GPU
!                   (one thread per pair, batches of pairs) and on the CPU, and reports the
!                   largest differences in W, e1b/e2a and the core-core term, with timings.

subroutine acc_update_nddo()
#ifdef _OPENACC
  use parameters_C, only : ad, alp, alpb, am, aq, dd, ddp, dorbs, guess1, guess2, guess3, iod, natorb, po, qq, tore, &
    xfac, par1, par2, par3, par4, par5, par6, par7, par8, par9, par10, par11, par12, par13, par14, par15, par16, &
    par17, par18, par19, par20, par21, par22, par23, par24, par25, par26, par27, par28, par29, par30, par31, par32, &
    par33, par34, par35, par36, par37, par38, par39, par40, par41, par42, par43, par44, par45, par46
  use mndod_C, only : ch, ind2, inddd, inddp, indexd, indpp, indx, isym
  use molkst_C, only : id, l_feather, method_am1, method_mndod, method_pm6, method_pm6_org, method_pm7, method_pm8, &
    mpack, trunc_1, trunc_2
  use funcon_C, only : a0, ev
  implicit none
  !$acc update device(ad, alp, alpb, am, aq, dd, ddp, dorbs, guess1, guess2, guess3, iod, natorb, po, qq, tore, xfac)
  !$acc update device(par1, par2, par3, par4, par5, par6, par7, par8, par9, par10, par11, par12, par13, par14, par15)
  !$acc update device(par16, par17, par18, par19, par20, par21, par22, par23, par24, par25, par26, par27, par28, par29)
  !$acc update device(par30, par31, par32, par33, par34, par35, par36, par37, par38, par39, par40, par41, par42, par43)
  !$acc update device(par44, par45, par46)
  !$acc update device(ch, ind2, inddd, inddp, indexd, indpp, indx, isym)
  !$acc update device(id, l_feather, method_am1, method_mndod, method_pm6, method_pm6_org, method_pm7, method_pm8)
  !$acc update device(mpack, trunc_1, trunc_2, a0, ev)
#endif
end subroutine acc_update_nddo

subroutine acc_integral_test()
#ifdef _OPENACC
  use molkst_C, only : numat
  use common_arrays_C, only : nat, coord
  use parameters_C, only : natorb, dorbs, iod
  use chanel_C, only : iw
  implicit none
  integer, parameter :: nbatch = 500000, nval = 121        ! 100 W + 10 e1b + 10 e2a + enuc (sp pairs only)
  integer :: npair, ip, i, j, k, b0, b1, nb, kr, li, lj, nw, nskip, stat
  integer, allocatable :: pi(:), pj(:), znat(:)
  double precision, allocatable :: xyz(:,:), vg(:,:)
  double precision :: wloc(2025), e1b(45), e2a(45), enuc, dw, de, dn, rw, t0, t1, tg, tc, tk, dsp
  double precision :: wsp(100), e1sp(10), e2sp(10), ensp
  logical :: spok
  character(len=16) :: env
  integer(8) :: c0, c1, rate
  !$acc routine(rotate) seq
  !$acc routine(rotate_sp) seq

  call get_environment_variable("MOPAC_ACC_INTEGRAL_TEST", env, status=stat)
  if (stat /= 0) return
  call acc_update_nddo()
  allocate (znat(numat), xyz(3, numat))
  znat = nat(1:numat)
  xyz = coord(:, 1:numat)
  npair = 0
  nskip = 0
  do i = 2, numat
    do j = 1, i - 1
      if (dorbs(znat(i)) .or. dorbs(znat(j)) .or. iod(znat(i)) > 0 .or. iod(znat(j)) > 0) then
        nskip = nskip + 1
      else
        npair = npair + 1
      end if
    end do
  end do
  allocate (pi(npair), pj(npair), vg(nval, nbatch))
  ip = 0
  do i = 2, numat
    do j = 1, i - 1
      if (dorbs(znat(i)) .or. dorbs(znat(j)) .or. iod(znat(i)) > 0 .or. iod(znat(j)) > 0) cycle
      ip = ip + 1
      pi(ip) = i
      pj(ip) = j
    end do
  end do
  dw = 0.d0; de = 0.d0; dn = 0.d0; rw = 0.d0; tg = 0.d0; tc = 0.d0; tk = 0.d0; dsp = 0.d0; spok = .true.
  !$acc data copyin(xyz, znat) create(vg)
  do b0 = 1, npair, nbatch
    b1 = min(npair, b0 + nbatch - 1)
    nb = b1 - b0 + 1
    call system_clock(c0, rate)
    !$acc parallel loop gang vector private(wsp, e1sp, e2sp, ensp, li, lj, nw, k) copyin(pi(b0:b1), pj(b0:b1)) &
    !$acc& present(xyz, znat, vg)
    do ip = 1, nb
      call rotate_sp(znat(pi(b0 + ip - 1)), znat(pj(b0 + ip - 1)), xyz(1, pi(b0 + ip - 1)), xyz(1, pj(b0 + ip - 1)), &
        wsp, e1sp, e2sp, ensp)
      li = natorb(znat(pi(b0 + ip - 1)))
      lj = natorb(znat(pj(b0 + ip - 1)))
      nw = ((li*(li + 1))/2)*((lj*(lj + 1))/2)
      do k = 1, 100
        vg(k, ip) = 0.d0
        if (k <= nw) vg(k, ip) = wsp(k)
      end do
      do k = 1, 10
        vg(100 + k, ip) = 0.d0
        vg(110 + k, ip) = 0.d0
        if (k <= (li*(li + 1))/2) vg(100 + k, ip) = e1sp(k)
        if (k <= (lj*(lj + 1))/2) vg(110 + k, ip) = e2sp(k)
      end do
      vg(121, ip) = ensp
    end do
    call system_clock(c1)
    tk = tk + dble(c1 - c0)/dble(rate)   ! kernel (the parallel loop is synchronous)
    !$acc update host(vg(:, 1:nb))
    call system_clock(c1)
    tg = tg + dble(c1 - c0)/dble(rate)
    call system_clock(c0)
    do ip = 1, nb
      i = pi(b0 + ip - 1)
      j = pj(b0 + ip - 1)
      kr = 1
      call rotate(znat(i), znat(j), xyz(1, i), xyz(1, j), wloc, kr, e1b, e2a, enuc)
      li = natorb(znat(i))
      lj = natorb(znat(j))
      nw = ((li*(li + 1))/2)*((lj*(lj + 1))/2)
      do k = 1, nw
        dw = max(dw, abs(vg(k, ip) - wloc(k)))
        rw = max(rw, abs(vg(k, ip) - wloc(k))/max(1.d-12, abs(wloc(k))))
      end do
      do k = 1, (li*(li + 1))/2
        de = max(de, abs(vg(100 + k, ip) - e1b(k)))
      end do
      do k = 1, (lj*(lj + 1))/2
        de = max(de, abs(vg(110 + k, ip) - e2a(k)))
      end do
      dn = max(dn, abs(vg(121, ip) - enuc))
! CPU rotate_sp must equal CPU rotate bitwise for these pairs.
      call rotate_sp(znat(i), znat(j), xyz(1, i), xyz(1, j), wsp, e1sp, e2sp, ensp)
      do k = 1, nw
        if (wsp(k) /= wloc(k)) spok = .false.
        dsp = max(dsp, abs(wsp(k) - wloc(k)))
      end do
      do k = 1, (li*(li + 1))/2
        if (e1sp(k) /= e1b(k)) spok = .false.
      end do
      do k = 1, (lj*(lj + 1))/2
        if (e2sp(k) /= e2a(k)) spok = .false.
      end do
      if (ensp /= enuc) spok = .false.
    end do
    call system_clock(c1)
    tc = tc + dble(c1 - c0)/dble(rate)
  end do
  !$acc end data
  t0 = tg; t1 = tc
  write (iw, '(/,a,i0,a,i0,a)') ' ACC INTEGRAL TEST: ', npair, ' sp atom pairs (', nskip, ' pairs with d atoms skipped)'
  write (iw, '(a,es10.3,a,es10.3,a,es10.3,a,es10.3)') ' ACC INTEGRAL TEST: max |dW| ', dw, '  max rel dW ', rw, &
    '  max |de1b,e2a| ', de, '  max |denuc| ', dn
  write (iw, '(a,l2,a,es10.3)') ' ACC INTEGRAL TEST: CPU rotate_sp bitwise equal to rotate:', spok, '  max |dW| ', dsp
  write (iw, '(a,f10.3,a,f10.3,a,f10.3,a)') ' ACC INTEGRAL TEST: GPU kernels ', tk, ' s; GPU incl. copy back ', t0, &
    ' s; CPU rotate ', t1, ' s (includes the comparison)'
  flush (iw)
  if (trim(env) == 'exit') stop 'MOPAC_ACC_INTEGRAL_TEST=exit: stopping after the integral test'
  deallocate (pi, pj, vg, znat, xyz)
#endif
end subroutine acc_integral_test

subroutine spd_integral_check()
! If MOPAC_ACC_INTEGRAL_TEST is set: for every atom pair in which exactly one atom has d orbitals
! (the preconditions of rotate_spd), compares rotate_spd with rotate on the CPU. They must be
! bitwise equal. Runs in every build (it is not OpenACC code).
  use molkst_C, only : numat, id
  use common_arrays_C, only : nat, coord
  use parameters_C, only : natorb, dorbs, iod
  use chanel_C, only : iw
  implicit none
  integer :: i, j, k, k2, kr, li, lj, nw, npair, nbad, stat, a, b
  double precision :: wloc(2025), e1b(45), e2a(45), enuc, wspd(450), e1s(45), e2s(45), ens, dmax
  logical :: ok, spa, spb
  character(len=16) :: env
  call get_environment_variable("MOPAC_ACC_INTEGRAL_TEST", env, status=stat)
  if (stat /= 0 .or. id /= 0) return
  npair = 0
  nbad = 0
  dmax = 0.d0
  do i = 2, numat
    do j = 1, i - 1
      do k = 1, 2
        if (k == 1) then
          a = i; b = j
        else
          a = j; b = i                     ! both orders: the d atom first and second
        end if
        spa = natorb(nat(a)) <= 4 .and. .not. dorbs(nat(a)) .and. iod(nat(a)) == 0
        spb = natorb(nat(b)) <= 4 .and. .not. dorbs(nat(b)) .and. iod(nat(b)) == 0
        if (.not. ((natorb(nat(a)) == 9 .and. spb) .or. (natorb(nat(b)) == 9 .and. spa))) cycle
        npair = npair + 1
        kr = 1
        call rotate(nat(a), nat(b), coord(1, a), coord(1, b), wloc, kr, e1b, e2a, enuc)
        call rotate_spd(nat(a), nat(b), coord(1, a), coord(1, b), wspd, e1s, e2s, ens)
        li = natorb(nat(a))
        lj = natorb(nat(b))
        nw = ((li*(li + 1))/2)*((lj*(lj + 1))/2)
        ok = (ens == enuc)
        do k2 = 1, nw
          if (wspd(k2) /= wloc(k2)) ok = .false.
          dmax = max(dmax, abs(wspd(k2) - wloc(k2)))
        end do
        do k2 = 1, (li*(li + 1))/2
          if (e1s(k2) /= e1b(k2)) ok = .false.
        end do
        do k2 = 1, (lj*(lj + 1))/2
          if (e2s(k2) /= e2a(k2)) ok = .false.
        end do
        if (.not. ok) nbad = nbad + 1
      end do
    end do
  end do
  write (iw, '(/,a,i0,a,i0,a,es10.3)') ' SPD INTEGRAL CHECK: ', npair, ' ordered s,p-d atom pairs; ', nbad, &
    ' not bitwise equal to rotate; max |dW| ', dmax
! rotate_sp_w (the integrals w only, for the GPU Fock build) against rotate, for every pair of s,p atoms
  npair = 0
  nbad = 0
  dmax = 0.d0
  do i = 2, numat
    do j = 1, i - 1
      spa = natorb(nat(i)) <= 4 .and. .not. dorbs(nat(i)) .and. iod(nat(i)) == 0
      spb = natorb(nat(j)) <= 4 .and. .not. dorbs(nat(j)) .and. iod(nat(j)) == 0
      if (.not. (spa .and. spb)) cycle
      npair = npair + 1
      kr = 1
      call rotate(nat(i), nat(j), coord(1, i), coord(1, j), wloc, kr, e1b, e2a, enuc)
      call rotate_sp_w(nat(i), nat(j), coord(1, i), coord(1, j), wspd)
      li = natorb(nat(i))
      lj = natorb(nat(j))
      nw = ((li*(li + 1))/2)*((lj*(lj + 1))/2)
      ok = .true.
      do k2 = 1, nw
        if (wspd(k2) /= wloc(k2)) ok = .false.
        dmax = max(dmax, abs(wspd(k2) - wloc(k2)))
      end do
      if (.not. ok) nbad = nbad + 1
    end do
  end do
  write (iw, '(a,i0,a,i0,a,es10.3)') ' SP_W INTEGRAL CHECK: ', npair, ' s,p atom pairs; ', nbad, &
    ' not bitwise equal to rotate; max |dW| ', dmax
  flush (iw)
end subroutine spd_integral_check
