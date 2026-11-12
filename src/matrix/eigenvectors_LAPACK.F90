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

    Subroutine eigenvectors_LAPACK(eigenvecs, xmat, eigvals, ndim)
      USE chanel_C, only : iw
      use mopac_magma, only : magma_fits, magma_calls, magma_cpu_fallbacks, magma_timing, magma_wall
#ifdef MOPAC_MAGMA
      use mopac_magma, only : magma_dsyevd, magma_dsyevd_m, magma_ngpu, MagmaUpper, MagmaLower, MagmaVec, &
        magma_workspace, mg_work, mg_iwork
#endif
#ifdef GPU
      Use mod_vars_cuda, only: lgpu, ngpus, prec
#endif

      implicit none
      Integer :: ndim
      double precision :: eigenvecs(ndim,ndim), &
                    & eigvals(ndim),xmat((ndim*(ndim+1))/2)
      integer :: i, j
      Integer :: lwork, liwork, info
      double precision,dimension(1:10) :: work_tmp
      Integer, dimension(1:10) :: iwork_tmp
      double precision, allocatable :: work(:)
      Integer, allocatable :: iwork(:)
      logical :: use_gpu, ok
      double precision :: t0, t1, t2, t3, t4
!==============================================================================
! Code to find all eigenvectors and all eigenvalues for a symmetric General matrix
! using LAPACK and MAGMA
! Gerd Bruno Rocha and Julio Carvalho Maia 11/17/2013.
! MAGMA path (-DMOPAC_MAGMA): magma_dsyevd when the matrix is large enough and fits in free
! GPU memory (module mopac_magma); on any MAGMA error the matrix is unpacked again from xmat
! and solved with CPU dsyevd.
!==============================================================================
      continue
      t0 = magma_wall()
      eigvals = 0.d0
      eigenvecs = 0.d0

!
! Perturb secular determinant matrix to split exact degeneracies
! (This is to get around a bug in the diagonalizer that causes eigenvectors to not be orthonormal)
!
!  The following unusual construction works.  Do NOT change it unless degeneracy tests have been done,
!  and the results are reproducible.
!
      forall (i=1:ndim)
          xmat((i*(i + 1))/2) = xmat((i*(i + 1))/2) + i*1.d-10   ! Do NOT go much higher than 1.d-10, otherwise the geometry
      endforall                                              ! optimization might go into an endless loop.
      call dtpttr( 'u', ndim, xmat, eigenvecs, ndim, i )

#ifdef MKL
#ifdef GPU
if (lgpu .and. (ngpus > 1 .and. ndim > 100)) then
      call mkl_dimatcopy('C', 'T' , ndim, ndim, 1.0d0, eigenvecs, ndim, ndim)
end if
#endif
#endif
      if (i /= 0) stop 'error in dtpttr'

      j = i ! Dummy - to make FORCHECK not complain about "j"
      i = j ! Dummy
      if (i == -999) return

      use_gpu = magma_fits(ndim)
#ifdef MOPAC_MAGMA
      if (use_gpu .and. magma_ngpu > 1) then
! Multi-GPU: MAGMA distributes the back-transformation only for the lower triangle, so mirror the
! upper triangle (filled by dtpttr) into the lower one and solve with MagmaLower.
        do j = 1, ndim
          eigenvecs(j+1:ndim, j) = eigenvecs(j, j+1:ndim)
        end do
      end if
      t1 = magma_wall()
      if (use_gpu) then
        if (magma_ngpu > 1) then
          i = magma_dsyevd_m(magma_ngpu, MagmaVec, MagmaLower, ndim, eigenvecs, ndim, eigvals, work_tmp, -1, &
            iwork_tmp, -1, info)
        else
          i = magma_dsyevd(MagmaVec, MagmaUpper, ndim, eigenvecs, ndim, eigvals, work_tmp, -1, iwork_tmp, -1, info)
        end if
        if (info == 0) then
          if (work_tmp(1) <= dble(huge(lwork))) then
            lwork = int(work_tmp(1))
            liwork = iwork_tmp(1)
            t2 = magma_wall()
            call magma_workspace(lwork, liwork, ok)
            t3 = magma_wall()
            if (ok) then
              if (magma_ngpu > 1) then
                i = magma_dsyevd_m(magma_ngpu, MagmaVec, MagmaLower, ndim, eigenvecs, ndim, eigvals, mg_work, lwork, &
                  mg_iwork, liwork, info)
              else
                i = magma_dsyevd(MagmaVec, MagmaUpper, ndim, eigenvecs, ndim, eigvals, mg_work, lwork, mg_iwork, liwork, info)
              end if
              t4 = magma_wall()
              if (magma_timing) write(iw,'(a,i0,a,4f8.2,a,f8.2)') ' MAGMA order ', ndim, &
                ': zero+perturb+unpack, query, workspace, solve (s):', t1 - t0, t2 - t1, t3 - t2, t4 - t3, &
                '  total', t4 - t0
              if (info == 0) then
                magma_calls = magma_calls + 1
                return
              end if
            else
              info = -1000
            end if
          else
            info = -1001
          end if
        end if
! MAGMA failed: restore the (already perturbed) matrix from xmat and use CPU LAPACK.
        magma_cpu_fallbacks = magma_cpu_fallbacks + 1
        write(iw,'(a,i0,a,i0,a)') ' MAGMA dsyevd failed (info = ', info, ') for order ', ndim, '; using CPU LAPACK'
        eigvals = 0.d0
        eigenvecs = 0.d0
        call dtpttr( 'u', ndim, xmat, eigenvecs, ndim, i )
      end if
#else
      if (use_gpu) write(iw,*) ' (unreachable: MAGMA not compiled in)'
#endif
      lwork = -1
      liwork = -1
      call dsyevd('v','u',ndim,eigenvecs,ndim,eigvals,work_tmp, &
                    & lwork,iwork_tmp,liwork,info)
! dsyevd needs lwork = 1 + 6N + 2N**2, which passes the 32-bit integer range near N = 32,768.
      if (work_tmp(1) > dble(huge(lwork))) then
        write(iw,'(a,i0,a,es10.3,a)') ' dsyevd workspace for order ', ndim, ' (', work_tmp(1), &
          ' doubles) exceeds the 32-bit LAPACK integer range'
        call mopend('Matrix too large for 32-bit LAPACK dsyevd workspace')
        return
      end if
      lwork = int(work_tmp(1))
      liwork = iwork_tmp(1)
      allocate (work(lwork), iwork(liwork), stat = i)
!      forall (j=1:lwork) work(j) = 0.d0
!      forall (j=1:liwork) iwork(j) = 0

      call dsyevd('v','u',ndim,eigenvecs,ndim,eigvals,work, &
                                & lwork,iwork,liwork,info)

      deallocate (iwork,work,stat = j)
      if (info /= 0)  write(iw,*) ' dsyevd Diagonalization error., CODE =',info
      continue
      return
    End subroutine eigenvectors_LAPACK
