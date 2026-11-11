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

      module funcon_C
      double precision :: &
      &  fpc_1,     & !
      &  fpc_2,     & !
      &  a0,        & !
      &  ev,        & !
      &  fpc_5,     & !
      &  fpc_6,     & !
      &  fpc_7,     & !
      &  fpc_8,     & !
      &  fpc_9,     & !
      &  fpc_10
      double precision, dimension (10) :: fpc = 0.d0
      double precision, parameter :: pi = 3.14159265358979323846d0
      double precision, parameter :: twopi = 2.0d0 * pi
! a0 and ev are not equivalenced into fpc: device copies (OpenACC declare) cannot alias another
! variable. They are set from fpc(3) and fpc(4) wherever fpc is assigned (readmo, mopac_api_initialize).
      equivalence (fpc(1), fpc_1), (fpc(2), fpc_2), &
      & (fpc(5), fpc_5), (fpc(6), fpc_6), (fpc(7), fpc_7), (fpc(8), fpc_8), &
      & (fpc(9), fpc_9), (fpc(10), fpc_10)
! Device copies for the OpenACC NDDO pair-integral routines (ignored without OpenACC).
!$acc declare create(a0, ev)
      end module funcon_C
