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
