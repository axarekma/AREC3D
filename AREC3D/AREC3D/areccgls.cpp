

#include "areccgls.h"
#include "arecConstants.h"
#include "arecImage.h"
#include "arecproject.h"
#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int cyl_cgls(MPI_Comm comm, arecImage images, float *angles, arecImage *xcvol, float lam, int maxit,
             float tol)
/*
 * CGLS: Conjugate Gradient for Least Squares
 *
 * Reference: A. Bjorck, "Numerical Methods for Least Squares Problems"
 * SIAM, 1996, pg. 289.
 * A method for solving large-scale, ill-posed inverse problems of the form:
 *             b = A*x + noise
 *
 */
{
    double t0;

    int ierr = 0, mypid, ncpus, status = 0;
    int height, radius;

    int nx, ny, nz, nangles, iter, i, j, nnz, iterbest;
    float *xcdata0 = nullptr;
    float *xcdata, *pdata, *gdata, *data, *prjdata, *sdata, *xcdata_save;

    arecImage gradvol, pdirvol, projstack, simages;

    double rnorm, rnorm0, rnorm2, rnorm2sum, relnrm, gamma, gamma0, alpha, beta, pnorm2, pnorm2sum,
        relnrm0, rnorm_min;

    MPI_Comm_rank(comm, &mypid);
    MPI_Comm_size(comm, &ncpus);

    nx = images.nx;
    ny = images.ny;
    nz = images.nz;
    data = images.data;
    nangles = nz;

    height = ny;
    radius = nx / 2;

    /* create temporary images to hold projections */
    status = arecAllocateCBImage(&projstack, nx, ny, nz);
    if (status != 0) {
        fprintf(stderr, "failed to allocate projstack\n");
        MPI_Abort(comm, ierr);
    }
    prjdata = projstack.data;
    for (i = 0; i < nx * ny * nz; i++)
        prjdata[i] = 0.0;

    /* temporary stack to hold the residuals */
    status = arecAllocateCBImage(&simages, nx, ny, nz);
    if (status != 0) {
        fprintf(stderr, "failed to allocate projstack\n");
        MPI_Abort(comm, ierr);
    }
    sdata = simages.data;

    /* temporary volume to hold the gradient images */
#ifdef DEBUG
    printf("mypid = %d, radius = %d, height = %d\n", mypid, radius, height);
#endif
    status = arecAllocateCylImage(&gradvol, radius, height);
    if (status != 0) {
        fprintf(stderr, "failed to allocate gradvol\n");
        MPI_Abort(comm, ierr);
    }
    gdata = gradvol.data;

    /* temporary volume to hold the conjugate direction */
    status = arecAllocateCylImage(&pdirvol, radius, height);
    if (status != 0) {
        fprintf(stderr, "failed to allocate pdirvol\n");
        MPI_Abort(comm, ierr);
    }
    pdata = pdirvol.data;

    /* if initial recontruction volume not provided,
       create one and initialize it to zero */
    if (xcvol->data == NULL) {
        /*debug*/
        printf("mypid = %d, null data\n", mypid);

        status = arecAllocateCylImage(xcvol, radius, height);
        if (status != 0) {
            fprintf(stderr, "failed to allocate xcvol\n");
            MPI_Abort(comm, ierr);
        }
        xcdata = xcvol->data;
        nnz = xcvol->nnz;
        for (i = 0; i < nnz; ++i)
            xcdata[i] = 0.0;
    } else {
        xcdata = xcvol->data;
        nnz = xcvol->nnz;
        status = arecProject2D(*xcvol, angles, nangles, &projstack);
    }

    xcdata_save = (float *)malloc(nnz * sizeof(float));

    for (i = 0; i < nx * ny * nz; i++)
        sdata[i] = data[i] - prjdata[i];

    if (mypid == 0) {
        printf("nrays = %d, nnz = %d\n", xcvol->nrays, nnz);
        printf("nx = %d, ny = %d, radius = %d, height = %d\n", nx, ny, radius, height);
    }

    iter = 1;
    iterbest = maxit - 1;
    t0 = MPI_Wtime();
    t0 = MPI_Wtime();

    gamma = 0.0;
    status = arecBackProject2D(simages, angles, nangles, &gradvol);
    if (status != 0) goto EXIT;

    /* calculate the norm of the backprojected volume */
    rnorm = 0.0;
    rnorm2 = 0.0;
    rnorm2sum = 0.0;
    for (j = 0; j < nnz; j++) {
        rnorm2 += gdata[j] * gdata[j];
    }
    MPI_Allreduce(&rnorm2, &rnorm2sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    gamma = rnorm2sum;
    rnorm0 = sqrt(rnorm2sum);
    if (mypid == 0) printf("CG iter = 0, resnrm = %11.3e\n", rnorm0);
    rnorm_min = rnorm0;

    xcdata0 = (float *)malloc(nnz * sizeof(float));
    while (iter <= maxit) {
        if (iter == 1) {
            for (j = 0; j < nnz; j++)
                pdata[j] = gdata[j];
        } else {
            beta = gamma / gamma0;
            for (j = 0; j < nnz; j++)
                pdata[j] = gdata[j] + beta * pdata[j];
            relnrm0 = relnrm;
        } /* end if (iter == 1) */

        status = arecProject2D(pdirvol, angles, nangles, &projstack);
        if (status != 0) goto EXIT;

        pnorm2 = 0.0;
        for (j = 0; j < nx * ny * nz; j++)
            pnorm2 += prjdata[j] * prjdata[j];

        pnorm2sum = 0.0;
        MPI_Allreduce(&pnorm2, &pnorm2sum, 1, MPI_DOUBLE, MPI_SUM, comm);

        alpha = gamma / pnorm2sum;

        for (j = 0; j < nnz; j++) {
            xcdata0[j] = xcdata[j];
            xcdata[j] = xcdata[j] + alpha * pdata[j];
        }

        for (j = 0; j < nx * ny * nz; j++)
            sdata[j] = sdata[j] - alpha * prjdata[j];

        status = arecBackProject2D(simages, angles, nangles, &gradvol);
        if (status != 0) goto EXIT;

        rnorm2 = 0.0;
        for (j = 0; j < nnz; j++) {
            rnorm2 += gdata[j] * gdata[j];
        }
        rnorm2sum = 0.0;
        MPI_Allreduce(&rnorm2, &rnorm2sum, 1, MPI_DOUBLE, MPI_SUM, comm);

        gamma0 = gamma;
        gamma = rnorm2sum;

        rnorm = sqrt(rnorm2sum);
        relnrm = rnorm / rnorm0;

        /* keep track of the best solution */
        if (rnorm < rnorm_min) {
            rnorm_min = rnorm;
            iterbest = iter;
            for (i = 0; i < nnz; i++)
                xcdata_save[i] = xcdata[i];
        }

        if (mypid == 0)
            printf("CG iter = %3d, rnorm / bnorm = %11.3e, rnorm = %11.3e\n", iter, relnrm, rnorm);

        //       if (iter > 1) {
        //          if (relnrm > relnrm0) {
        //             /* backtrack */
        //             for (i = 0; i<nnz; i++) xcdata[i] = xcdata0[i];
        //          }
        //          if (mypid == 0) {
        //             printf("Terminating with an approximation obtained at iter %d\n",
        //                    iter-1);
        //          }
        //          break;
        //       }

        /* if changes are sufficiently small, or if no further
                           progress is made, terminate */
        if (relnrm < tol) {
            if (mypid == 0) {
                printf("Terminating with rnorm/bnorm = %11.3e, tol = %11.3e, ", relnrm, tol);
            }
            break;
        }

        ++iter;
    }

    /* retrieve the best approximation */
    if (iterbest < maxit - 1) {
        if (mypid == 0) printf("retrieve the best solution\n");
        for (i = 0; i < nnz; i++)
            xcdata[i] = xcdata_save[i];
    }

    if (mypid == 0) printf("Total time in CG = %11.3e\n", MPI_Wtime() - t0);
    free(xcdata_save);
EXIT:

    free(xcdata0);
    arecImageFree(&pdirvol);
    arecImageFree(&gradvol);
    arecImageFree(&projstack);
    arecImageFree(&simages);

    return status;
}

int cyl_cgls_SQ(MPI_Comm comm, arecImage images, float *angles, arecImage *xcvol, float lam,
                int maxit, float tol)
/*
 * CGLS: Conjugate Gradient for Least Squares
 *
 * Reference: A. Bjorck, "Numerical Methods for Least Squares Problems"
 * SIAM, 1996, pg. 289.
 * A method for solving large-scale, ill-posed inverse problems of the form:
 *             b = A*x + noise
 *
 */
{
    double t0;

    int ierr = 0, mypid, ncpus, status = 0;
    int height, radius;

    int nx, ny, nz, nangles, iter, i, j, nnz, iterbest;
    float *xcdata0 = nullptr;
    float *xcdata, *pdata, *gdata, *data, *prjdata, *sdata, *xcdata_save;

    arecImage gradvol, pdirvol, projstack, simages;

    double rnorm, rnorm0, rnorm2, rnorm2sum, relnrm, gamma, gamma0, alpha, beta, pnorm2, pnorm2sum,
        relnrm0, rnorm_min;

    MPI_Comm_rank(comm, &mypid);
    MPI_Comm_size(comm, &ncpus);

    nx = images.nx;
    ny = images.ny;
    nz = images.nz;
    data = images.data;
    nangles = nz;

    height = ny;
    radius = nx / 2;

    /* create temporary images to hold projections */
    status = arecAllocateCBImage(&projstack, nx, ny, nz);
    if (status != 0) {
        fprintf(stderr, "failed to allocate projstack\n");
        MPI_Abort(comm, ierr);
    }
    prjdata = projstack.data;
    for (i = 0; i < nx * ny * nz; i++)
        prjdata[i] = 0.0;

    /* temporary stack to hold the residuals */
    status = arecAllocateCBImage(&simages, nx, ny, nz);
    if (status != 0) {
        fprintf(stderr, "failed to allocate projstack\n");
        MPI_Abort(comm, ierr);
    }
    sdata = simages.data;

    /* temporary volume to hold the gradient images */
#ifdef DEBUG
    // printf("mypid = %d, radius = %d, height = %d\n", mypid, radius, height);
#endif
    status = arecAllocateCylImage(&gradvol, radius, height);
    if (status != 0) {
        fprintf(stderr, "failed to allocate gradvol\n");
        MPI_Abort(comm, ierr);
    }
    gdata = gradvol.data;

    /* temporary volume to hold the conjugate direction */
    status = arecAllocateCylImage(&pdirvol, radius, height);
    if (status != 0) {
        fprintf(stderr, "failed to allocate pdirvol\n");
        MPI_Abort(comm, ierr);
    }
    pdata = pdirvol.data;

    /* if initial recontruction volume not provided,
       create one and initialize it to zero */
    if (xcvol->data == NULL) {
        /*debug*/
        printf("mypid = %d, null data\n", mypid);

        status = arecAllocateCylImage(xcvol, radius, height);
        if (status != 0) {
            fprintf(stderr, "failed to allocate xcvol\n");
            MPI_Abort(comm, ierr);
        }
        xcdata = xcvol->data;
        nnz = xcvol->nnz;
        for (i = 0; i < nnz; ++i)
            xcdata[i] = 0.0;
    } else {
        xcdata = xcvol->data;
        nnz = xcvol->nnz;
        status = arecProject2D_SQ(*xcvol, angles, nangles, &projstack);
    }

    xcdata_save = (float *)malloc(nnz * sizeof(float));

    for (i = 0; i < nx * ny * nz; i++) {
        sdata[i] = data[i] - prjdata[i];
    }

    if (mypid == 0) {
        printf("cyl: nrays = %d, nnz = %d\n", xcvol->nrays, nnz);
        printf("nx = %d, ny = %d, radius = %d, height = %d\n", nx, ny, radius, height);
    }

    iter = 1;
    iterbest = maxit - 1;
    t0 = MPI_Wtime();

    gamma = 0.0;
    status = arecBackProject2D_SQ(simages, angles, nangles, &gradvol);
    if (status != 0) goto EXIT;

    /* calculate the norm of the backprojected volume */
    rnorm = 0.0;
    rnorm2 = 0.0;
    rnorm2sum = 0.0;
    for (j = 0; j < nnz; j++) {
        rnorm2 += gdata[j] * gdata[j];
    }
    MPI_Allreduce(&rnorm2, &rnorm2sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    gamma = rnorm2sum;
    rnorm0 = sqrt(rnorm2sum);
    if (mypid == 0) printf("CG iter = 0, resnrm = %11.3e\n", rnorm0);
    rnorm_min = rnorm0;

    xcdata0 = (float *)malloc(nnz * sizeof(float));
    while (iter <= maxit) {
        if (iter == 1) {
            for (j = 0; j < nnz; j++)
                pdata[j] = gdata[j];
        } else {
            beta = gamma / gamma0;
            for (j = 0; j < nnz; j++)
                pdata[j] = gdata[j] + beta * pdata[j];
            relnrm0 = relnrm;
        } /* end if (iter == 1) */

        status = arecProject2D_SQ(pdirvol, angles, nangles, &projstack);
        if (status != 0) goto EXIT;

        pnorm2 = 0.0;
        for (j = 0; j < nx * ny * nz; j++)
            pnorm2 += prjdata[j] * prjdata[j];

        pnorm2sum = 0.0;
        MPI_Allreduce(&pnorm2, &pnorm2sum, 1, MPI_DOUBLE, MPI_SUM, comm);

        alpha = gamma / pnorm2sum;

        for (j = 0; j < nnz; j++) {
            xcdata0[j] = xcdata[j];
            xcdata[j] = xcdata[j] + alpha * pdata[j];
        }

        for (j = 0; j < nx * ny * nz; j++)
            sdata[j] = sdata[j] - alpha * prjdata[j];

        status = arecBackProject2D_SQ(simages, angles, nangles, &gradvol);
        if (status != 0) goto EXIT;

        rnorm2 = 0.0;
        for (j = 0; j < nnz; j++) {
            rnorm2 += gdata[j] * gdata[j];
        }
        rnorm2sum = 0.0;
        MPI_Allreduce(&rnorm2, &rnorm2sum, 1, MPI_DOUBLE, MPI_SUM, comm);

        gamma0 = gamma;
        gamma = rnorm2sum;

        rnorm = sqrt(rnorm2sum);
        relnrm = rnorm / rnorm0;

        /* keep track of the best solution */
        if (rnorm < rnorm_min) {
            rnorm_min = rnorm;
            iterbest = iter;
            for (i = 0; i < nnz; i++)
                xcdata_save[i] = xcdata[i];
        }

        if (mypid == 0)
            printf("CG iter = %3d, rnorm / bnorm = %11.3e, rnorm = %11.3e\n", iter, relnrm, rnorm);

        //       if (iter > 1) {
        //          if (relnrm > relnrm0) {
        //             /* backtrack */
        //             for (i = 0; i<nnz; i++) xcdata[i] = xcdata0[i];
        //          }
        //          if (mypid == 0) {
        //             printf("Terminating with an approximation obtained at iter %d\n",
        //                    iter-1);
        //          }
        //          break;
        //       }

        /* if changes are sufficiently small, or if no further
                           progress is made, terminate */
        if (relnrm < tol) {
            if (mypid == 0) {
                printf("Terminating with rnorm/bnorm = %11.3e, tol = %11.3e, ", relnrm, tol);
            }
            break;
        }

        ++iter;
    }

    /* retrieve the best approximation */
    if (iterbest < maxit - 1) {
        if (mypid == 0) printf("retrieve the best solution\n");
        for (i = 0; i < nnz; i++)
            xcdata[i] = xcdata_save[i];
    }

    if (mypid == 0) printf("Total time in CG = %11.3e\n", MPI_Wtime() - t0);
    free(xcdata_save);

EXIT:

    free(xcdata0);
    free(pdirvol.cord);
    free(pdirvol.data);
    arecImageFree(&gradvol);
    arecImageFree(&projstack);
    arecImageFree(&simages);

    return status;
}
