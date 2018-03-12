#include "fftw3.h"
#include <algorithm>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ReadAndDistAngles.h"
#include "align2dstack.h"
#include "arec3dutil.h"
#include "arecConstants.h"
#include "arecImage.h"
#include "arecImageIO.h"
#include "arecImageIO_mpi.h"
#include "areccgls.h"
#include "areccgls_KB.h"
#include "areccgls_stat.h"
#include "arecproject.h"
#include "arecsirt.h"
#include "arecsirt_KB.h"
#include "imagetools.h"

int main(int argc, char **argv) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int ncpus, mypid, status = 0;
    int nx, ny, nz, nyloc, nzloc, nangles, maxsirt, height, i, j, xcent, ycent, maxit, iter, lcut,
        rcut, fudgefactor, r2, nimgs, nimgstot, rmethod, iflip, pmethod;
    double radius;
    float delang, tol, lam;
    float *rotangles, *angbuf, *rotangles_sum;
    float *sx, *sy, *sx_temp, *sy_temp, *sxbuf, *sybuf;
    float *angles = nullptr;
    arecparam inparam;
    double t0;
    FILE *fp;
    char logfname[200];
    MPI_Status mpistatus;

    /* image data */
    arecImage expimages, alignedimages, xcyvol, xcbvol;
    arecImage prjstack;
    int rotsize;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &ncpus);
    MPI_Comm_rank(comm, &mypid);

#pragma omp parallel
    {
        if (omp_get_thread_num() == 0 && mypid == 0) {
            printf("\nMPI: Program running with world_size=%d "
                   "num_threads=%d\n\n",
                   ncpus, omp_get_num_threads());
        }
    }

    char inputfname[200], voutfname[200], prjstackfname[200], alignedimagesfname[200];

    // parse the command line and set filenames
    if (argc != 2) {
        if (mypid == 0) print_hint();
        status = 1;
        goto EXIT;
    }

#pragma warning(suppress : 4996)
    strcpy(inputfname, argv[1]);
    status = parseinput(comm, inputfname, &inparam);
    if (status != 0) {
        fprintf(stderr, "failed to parse the input file\n");
        goto EXIT;
    }

    // read and distribute a stack of experimental images
    t0 = MPI_Wtime();
    arecReadImageDistbyZ(comm, inparam.stackfname, &expimages);
    if (mypid == 0) { printf("I/O time for reading image stack = %11.3e\n", MPI_Wtime() - t0); }
    arecReadImageDistbyZ(comm, inparam.stackfname, &alignedimages);
    arecImageTakeLog(alignedimages);

    nx = expimages.nx;
    ny = expimages.ny;
    nzloc = expimages.nz;
    MPI_Allreduce(&nzloc, &nimgstot, 1, MPI_INT, MPI_SUM, comm);
#ifdef DEBUG
    printf("mypid = %d, image size: nx = %d, ny = %d, nzloc = %d\n", mypid, nx, ny, nzloc);
#endif

    /* read and distribute angles if an angle file is available */
    if (inparam.haveangles) {
        angles = (float *)calloc(nimgstot, sizeof(float));
        status = ReadAndDistAngles(comm, inparam.anglefname, nimgstot, angles, &iflip);
        if (status != 0) goto EXIT;
    } else {
        iflip = 90;
    }
    if (mypid == 0) printf("iflip = %d\n", iflip);

    /* set parameter based on input */
    setparams(inparam, nx, ny, &radius, &height, &xcent, &ycent, voutfname, &lcut, &rcut,
              &fudgefactor, &rmethod, &pmethod);

    periodicDecomposition(alignedimages);
    /* write the stack out as a single file */
    arecWriteImageMergeZ(comm, "alignedimages_iter0.mrc", alignedimages);
    arecImageFree(&alignedimages);

    // BEGIN INITIAL RECONSTRUCTION //
    /* read the aligned and cropped stack back and distribute
    each image along Y direction */
    arecReadImageDistbyY(comm, "alignedimages_iter0.mrc", &alignedimages);

    nx = alignedimages.nx;
    nyloc = alignedimages.ny; /* local ny */
    nz = alignedimages.nz;
#ifdef DEBUG
    printf("mypid = %d, image size: nx = %d, nyloc = %d, nz = %d\n", mypid, nx, nyloc, nz);
#endif

    /* set up the angles */
    nangles = nz;
    if (!inparam.haveangles) {
        delang = 2.0 * piFunc() / 180.0;
        angles = (float *)malloc(nangles * sizeof(float));
        for (i = 0; i < nangles; i++)
            angles[i] = i * delang;
    }

    lam = inparam.lam;
    maxsirt = inparam.maxsirt;
    tol = inparam.tolsirt;
    if (mypid == 0) {
        if (rmethod == 1) {
            printf("lam = %11.3e, maxsirt = %d, tol = %11.3e\n", lam, maxsirt, tol);
        } else {
            printf("maxcgiter = %d, tol = %11.3e\n", maxsirt, tol);
        }
    }
    /* do an initial reconstruction */
    /* xcyvol has not been created yet (this is a bit tricky) */
    xcyvol.data = NULL;
    // printf("Reconstruction using rmethod:%d pmethod:%d\n",rmethod,pmethod);
    if (rmethod == 1) {
        if (pmethod == 1)
            status = cyl_sirt(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
        if (pmethod == 2)
            status = cyl_sirt_KB(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
        if (pmethod == 3)
            status = cyl_sirt_SQ(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
    } else if (rmethod == 2) {
        if (pmethod == 1)
            status = cyl_cgls(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
        if (pmethod == 2)
            status = cyl_cgls_KB(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
        if (pmethod == 3)
            status = cyl_cgls_SQ(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
    } else if (rmethod == 3) {
        if (pmethod == 1)
            status = cyl_cgls_stat(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
        if (pmethod == 2)
            status = cyl_cgls_KB_stat(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
        if (pmethod == 3)
            status = cyl_cgls_SQ_stat(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
    } else {
        fprintf(stderr, "Invalid reconstruction method: %d\n", rmethod);
        goto EXIT;
    }
    arecImageFree(&alignedimages);
    // END INITIAL RECONSTRUCTION //
    if (mypid == 0) printf("Initial reconstruction is done\n");
    //  now do projection matching
    sx = (float *)calloc(nzloc, sizeof(float));
    sy = (float *)calloc(nzloc, sizeof(float));
    sx_temp = (float *)calloc(nzloc, sizeof(float));
    sy_temp = (float *)calloc(nzloc, sizeof(float));
    int corr_iter;

    maxit = inparam.maxit;
    rotangles = (float *)calloc(nzloc, sizeof(float));
    rotangles_sum = (float *)calloc(nzloc, sizeof(float));

    for (iter = 0; iter < maxit; iter++) {
        if (mypid == 0) printf("AREC projection matching iter %d \n", iter);
        // generate projections the reconstruction
        arecAllocateCBImage(&prjstack, nx, nyloc, nangles);
        if (pmethod == 1) arecProject2D(xcyvol, angles, nangles, &prjstack);
        if (pmethod == 2) arecProject2D_KB(xcyvol, angles, nangles, &prjstack);
        if (pmethod == 3) arecProject2D_SQ(xcyvol, angles, nangles, &prjstack);
        MPI_Barrier(comm);
#pragma warning(suppress : 4996)
        sprintf(prjstackfname, "projected_stack.mrc");
        arecWriteImageMergeY(comm, prjstackfname, prjstack);
        arecImageFree(&prjstack);
        // Read both the projected data and the original images (iter 0 is not shifted but
        // preprocessed)  and distribute them by angles
        arecReadImageDistbyZ(comm, prjstackfname, &prjstack);
        arecReadImageDistbyZ(comm, "alignedimages_iter0.mrc", &alignedimages);
        // rotate original image
        if (iter > 0) arecRotateImages_skew(&alignedimages, rotangles_sum);

        // iteratively find best translation
        for (i = 0; i < nzloc; i++) {
            sx[i] = 0.0;
            sy[i] = 0.0;
        }
        if (mypid == 0) printf("Translational correlation\n");
        // TODO change to while loop, check for sx_temp sy_temp
        for (corr_iter = 0; corr_iter < 3; corr_iter++) {
            status = arecCCImages(comm, prjstack, alignedimages, sx_temp, sy_temp);
            if (status != 0) goto EXIT;
            if (mypid == 0) printf("arecCC done\n");
            arecShiftImages(alignedimages, sx_temp, sy_temp);
            for (i = 0; i < nzloc; i++) {
                sx[i] += sx_temp[i];
                sy[i] += sy_temp[i];
            }
        }

        // load fresh data and transform with current transformations
        arecReadImageDistbyZ(comm, "alignedimages_iter0.mrc", &alignedimages);
        if (iter > 0) arecRotateImages_skew(&alignedimages, rotangles_sum);
        arecShiftImages(alignedimages, sx, sy);

        // do rotational correlation
        // only do rotational correlation every 3rd time to ensure trans is good.
        if ((iter + 1) % 3 == 0) {
            if (mypid == 0) printf("iter==%d, doing rotcorr\n", iter);
            rotsize = std::min(static_cast<int>(2 * radius + 1), height - 10);
            r2 = rotsize / 2;
            status = arecRotCCImages(comm, prjstack, alignedimages, r2, rotangles);
            arecRotateImages_skew_safe(&alignedimages, &prjstack, rotangles);
            for (i = 0; i < nzloc; i++) {
                rotangles_sum[i] = rotangles_sum[i] + rotangles[i];
            }
        }

        MPI_Barrier(comm);
#pragma warning(suppress : 4996)
        sprintf(alignedimagesfname, "alignedimages_iter.mrc");
        arecWriteImageMergeZ(comm, alignedimagesfname, alignedimages);
        arecImageFree(&alignedimages);
        arecImageFree(&prjstack);

        if (mypid == 0) {
            /* write shifts and rotation angle to the LOG file */
#pragma warning(suppress : 4996)
            sprintf(logfname, "arec3dLOG.iter%d", iter + 1);
#pragma warning(suppress : 4996)
            fp = fopen(logfname, "wb");
            sxbuf = (float *)malloc(nzloc * sizeof(float));
            sybuf = (float *)malloc(nzloc * sizeof(float));
            angbuf = (float *)malloc(nzloc * sizeof(float));
            for (i = 0; i < ncpus; i++) {
                if (i == 0) {
                    nimgs = nzloc;
                    for (j = 0; j < nimgs; j++) {
                        sxbuf[j] = sx[j];
                        sybuf[j] = sy[j];
                        angbuf[j] = rotangles_sum[j];
                    }
                } else {
                    /* Receive from proc i */
                    MPI_Recv(&nimgs, 1, MPI_INT, i, 0, comm, &mpistatus);
                    MPI_Recv(sxbuf, nimgs, MPI_FLOAT, i, 1, comm, &mpistatus);
                    MPI_Recv(sybuf, nimgs, MPI_FLOAT, i, 2, comm, &mpistatus);
                    MPI_Recv(angbuf, nimgs, MPI_FLOAT, i, 3, comm, &mpistatus);
                }
                for (j = 0; j < nimgs; j++)
                    fprintf(fp, "%11.4e  %11.4e  %11.4e\n", sxbuf[j], sybuf[j], angbuf[j]);
            } /* end for i */
            fclose(fp);
            free(sxbuf);
            free(sybuf);
            free(angbuf);
        } else {
            MPI_Send(&nzloc, 1, MPI_INT, 0, 0, comm);
            MPI_Send(sx, nzloc, MPI_FLOAT, 0, 1, comm);
            MPI_Send(sy, nzloc, MPI_FLOAT, 0, 2, comm);
            MPI_Send(rotangles_sum, nzloc, MPI_FLOAT, 0, 3, comm);
        }

        MPI_Barrier(comm);

        // read the aligned images back in and distribute them by Y
        arecReadImageDistbyY(comm, alignedimagesfname, &alignedimages);
        if (rmethod == 1) {
            if (pmethod == 1)
                status = cyl_sirt(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
            if (pmethod == 2)
                status = cyl_sirt_KB(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
            if (pmethod == 3)
                status = cyl_sirt_SQ(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
        } else if (rmethod == 2) {
            if (pmethod == 1)
                status = cyl_cgls(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
            if (pmethod == 2)
                status = cyl_cgls_KB(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
            if (pmethod == 3)
                status = cyl_cgls_SQ(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
        } else if (rmethod == 3) {
            if (pmethod == 1)
                status = cyl_cgls_stat(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
            if (pmethod == 2)
                status = cyl_cgls_KB_stat(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
            if (pmethod == 3)
                status = cyl_cgls_SQ_stat(comm, alignedimages, angles, &xcyvol, lam, maxsirt, tol);
        }

        // suppress negative between CG iterations
        soft_noneg(&xcyvol);

        MPI_Barrier(comm);

        arecImageFree(&alignedimages);
        arecImageFree(&prjstack);

    } /* end for iter */

    /* turn the reconstruction into cubic format before writing it out */
    arecAllocateCBImage(&xcbvol, xcyvol.nx, xcyvol.ny, xcyvol.nz);
    ImageCyl2CB(xcyvol, xcbvol, xcyvol.nx, xcyvol.ny, xcyvol.nz);
    arecWriteImageMergeY(comm, voutfname, xcbvol);
    arecImageFree(&xcyvol);

    arecImageFree(&xcbvol);
    arecImageFree(&expimages);

    free(angles);
    free(rotangles);
    free(rotangles_sum);

    free(sx);
    free(sy);
    free(sx_temp);
    free(sy_temp);

EXIT:
    MPI_Finalize();
    return status;
}
