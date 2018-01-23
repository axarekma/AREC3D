
#include "arecsirt_KB.h"
#include "arecConstants.h"
#include "arecImage.h"
#include "arecproject.h"
#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int cyl_sirt_KB(MPI_Comm comm, arecImage images, float *angles, arecImage *xcvol, float lam,
                int maxit, float tol) {
    double t0;

    int ierr = 0, mypid, ncpus, status = 0;
    int height, radius;

    int nx, ny, nz, nangles, iter, i, j, nnz;
    float *xcdata, *pdata, *bdata, *grad;

    arecImage bvol, pcylvol, projstack;

    double rnorm, rnorm2, rnorm2sum, bnorm, bnorm2, bnorm2sum;
    double relnrm, relnrm0;
    int restarts = 0;

    MPI_Comm_rank(comm, &mypid);
    MPI_Comm_size(comm, &ncpus);

    nx = images.nx;
    ny = images.ny;
    nz = images.nz;
    nangles = nz;

    height = ny;
    radius = nx / 2;

    /* create volume image and initialize it to zero */
    status = arecAllocateCBImage(&projstack, nx, ny, nz);
    if (status != 0) {
        fprintf(stderr, "failed to allocate projstack\n");
        MPI_Abort(comm, ierr);
    }

    if (xcvol->data == NULL) {
        status = arecAllocateCylImage(xcvol, radius, height);
        if (status != 0) {
            fprintf(stderr, "failed to allocate xcvol\n");
            MPI_Abort(comm, ierr);
        }
    }
    xcdata = xcvol->data;
    nnz = xcvol->nnz;
    for (i = 0; i < nnz; ++i)
        xcdata[i] = 0.0;
    if (mypid == 0) {
        printf("nrays = %d, nnz = %d\n", xcvol->nrays, nnz);
        printf("nx = %d, ny = %d, radius = %d, height = %d\n", nx, ny, radius, height);
    }

    status = arecAllocateCylImage(&bvol, radius, height);
    if (status != 0) {
        fprintf(stderr, "failed to allocate bvol\n");
        MPI_Abort(comm, ierr);
    }
    bdata = bvol.data;

    status = arecAllocateCylImage(&pcylvol, radius, height);
    if (status != 0) {
        fprintf(stderr, "failed to allocate pcylvol\n");
        MPI_Abort(comm, ierr);
    }
    pdata = pcylvol.data;

    /* kluge, make sure if its 1.0 + epsilon it still works */
    relnrm0 = 1.00001;

    grad = (float *)calloc(nnz, sizeof(float));
    if (!grad) {
        fprintf(stderr, "failed to allocate work arrays in sirt\n");
        MPI_Abort(comm, ierr);
    }

    iter = 1;

    rnorm = 0.0;
    rnorm2 = 0.0;
    rnorm2sum = 0.0;
    bnorm = 0.0;
    bnorm2 = 0.0;
    bnorm2sum = 0.0;

    t0 = MPI_Wtime();
    while (iter <= maxit) {
        if (iter == 1) {
            if (restarts == 0) {
                status = arecBackProject2D_KB(images, angles, nangles, &bvol);
                if (status != 0) goto EXIT;
            }

            /* calculate the norm of the backprojected volume */
            for (j = 0; j < nnz; j++) {
                bnorm2 += bdata[j] * bdata[j];
                grad[j] = bdata[j];
            }
            MPI_Allreduce(&bnorm2, &bnorm2sum, 1, MPI_DOUBLE, MPI_SUM, comm);
            bnorm = sqrt(bnorm2sum);
        } else {
            status = arecProject2D_KB(*xcvol, angles, nangles, &projstack);
            if (status != 0) goto EXIT;

            status = arecBackProject2D_KB(projstack, angles, nangles, &pcylvol);
            if (status != 0) goto EXIT;

            for (j = 0; j < nnz; j++) {
                grad[j] = bdata[j];
                grad[j] -= pdata[j];
            }
        } /* end if (iter == 1) */

        rnorm2 = 0.0;
        for (j = 0; j < nnz; j++) {
            rnorm2 += grad[j] * (double)grad[j];
        }
        rnorm2sum = 0.0;
        MPI_Allreduce(&rnorm2, &rnorm2sum, 1, MPI_DOUBLE, MPI_SUM, comm);
        rnorm = sqrt(rnorm2sum);
        relnrm = rnorm / bnorm;

        if (mypid == 0)
            printf("iter = %3d, rnorm / bnorm = %11.3e, rnorm = %11.3e\n", iter, relnrm, rnorm);
        /* if on the second pass, rnorm is greater than bnorm,
           lam is probably set too high reduce it by a factor of 2
           and start over */
        if (relnrm > relnrm0) {
            /* but don't do it more than 20 times */
            if (restarts > 20) {
                if (mypid == 0) printf("Failure to converge, even with lam = %11.3e\n", lam);
                break;
            } else {
                ++restarts;
                iter = 1;
                lam /= 2.0;
                /* reset these */
                /* kluge, make sure if its 1.0 + epsilon it still
                   works */
                relnrm0 = 1.0001;
                for (j = 0; j < nnz; ++j) {
                    xcdata[j] = 0.0;
                    pdata[j] = 0.0;
                }
                if (mypid == 0) printf("reducing lam to %11.3e, restarting\n", lam);
                continue;
            } /* end if (restarts > 20) */
        }     /* end if (rnorm/bnorm > relnrm0) */

        /* if changes are sufficiently small, or if no further
           progress is made, terminate */
        if (relnrm < tol || relnrm > relnrm0) {
            if (mypid == 0) {
                printf("Terminating with rnorm/bnorm = %11.3e, tol = %11.3e, ", relnrm, tol);
                printf("relnrm0 = %11.3e\n", relnrm0);
            }
            break;
        }

        /* update the termination threshold */
        relnrm0 = relnrm;

        /* update the reconstructed volume */
        for (j = 0; j < nnz; ++j) {
            xcdata[j] += lam * grad[j];
            /* reset it so it's ready to accumulate for the next
               iteration */
            pdata[j] = 0.0;
        }

        ++iter;
    }
    if (mypid == 0) printf("Total time in SIRT = %11.3e\n", MPI_Wtime() - t0);

EXIT:

    free(grad);
    arecImageFree(&pcylvol);
    arecImageFree(&bvol);
    arecImageFree(&projstack);

    return status;
}
