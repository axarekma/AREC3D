#include "fftw3.h"
#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "align2dstack.h"
#include "arecConstants.h"
#include "arecImage.h"
#include "arecImageIO.h"
#include "arecImageIO_mpi.h"

/* This is a bit confusing, imagedata is in FORTRAN style
   column major with ny rows and nx columns.
   But the images is considered row major with
   nx rows and ny columns when FFTW is used,
   so sy is applied to the fastest changing dimension (y),
   and sx is applied to the slowest changing dimension (x) */

#define imagedata(i, j, k) imagedata[nx * ny * (k) + nx * (j) + (i)]
#define croppeddata(i, j, k) croppeddata[tubewidth * ny * (k) + tubewidth * (j) + (i)]

#define EDGETHRESH 0.5
#define BACKTHRESH 0.9
#define CROPTHRESH 0.35

/* Align a stack of 2-D images using cross correlation */

void align2dstack(MPI_Comm comm, arecImage imagestack, arecImage alignedstack, int lcut, int rcut,
                  int fudgefactor, int iflip) {
    int ncpus, mypid, ierr = 0;
    int nx, ny, nzloc, nzsum, imgnum;
    int *psize, *nbase;
    int iloc, sid;
    int printid = 0;

    // image data
    float *imagedata, *croppeddata, *aligneddata, *imagebuf, *imageflip;
    float *cccoefs, *prj1d;
    float *relsx, *relsy, *abssx, *abssy, *abssx_crop, *abssy_crop, *relsx_crop, *relsy_crop;
    float sx, sy;
    int *edgeleft, *edgeright;
    float *sxbuf, *sybuf;
    int ibeg, iend, i, j, tubewidth, tubewidthloc, slack, nimgs;
    arecImage croppedimages;
    FILE *fp;

    MPI_Status mpistatus;
    MPI_Comm_size(comm, &ncpus);
    MPI_Comm_rank(comm, &mypid);
    /* printf("mypid = %d, ncpus = %d\n", mypid, ncpus); */

    /* align image imgnum with the previous image */
    imagedata = imagestack.data;
    nx = imagestack.nx;
    ny = imagestack.ny;
    nzloc = imagestack.nz;
    MPI_Allreduce(&nzloc, &nzsum, 1, MPI_INT, MPI_SUM, comm);

    cccoefs = (float *)fftwf_malloc(nx * ny * sizeof(float));
    abssx = (float *)calloc(nzloc, sizeof(float));
    abssy = (float *)calloc(nzloc, sizeof(float));
    relsx = (float *)calloc(nzloc, sizeof(float));
    relsy = (float *)calloc(nzloc, sizeof(float));
    relsx_crop = (float *)calloc(nzloc, sizeof(float));
    relsy_crop = (float *)calloc(nzloc, sizeof(float));
    abssx_crop = (float *)calloc(nzloc, sizeof(float));
    abssy_crop = (float *)calloc(nzloc, sizeof(float));

    imagebuf = (float *)fftwf_malloc(nx * ny * sizeof(float));
    imageflip = (float *)fftwf_malloc(nx * ny * sizeof(float));

    /* thresholding to clean out the background */
    for (imgnum = 0; imgnum < nzloc; imgnum++)
        for (j = 0; j < ny; j++)
            for (i = 0; i < nx; i++)
                if (imagedata(i, j, imgnum) > BACKTHRESH) imagedata(i, j, imgnum) = BACKTHRESH;

    /* align with the next image */
    for (imgnum = 1; imgnum < nzloc; imgnum++) {
        ccorr2d(nx, ny, &imagedata[nx * ny * (imgnum - 1)], &imagedata[nx * ny * imgnum], cccoefs);
        /* peak search */
        /* peaksearch(nx, ny, cccoefs, &relsx[imgnum], &relsy[imgnum]); */
        speak(nx, ny, cccoefs, &relsx[imgnum], &relsy[imgnum]);
    }

    if (mypid == 0) {
        relsx[0] = 0.0;
        relsy[0] = 0.0;
        /* send the last image to the next processor */
        if (ncpus > 1) {
            MPI_Send(&imagedata[nx * ny * (nzloc - 1)], nx * ny, MPI_FLOAT, 1, 0, comm);
        }
    } else if (mypid < nzsum) {
        /* receive an image from the previous processor to align
           the first image */
        MPI_Recv(imagebuf, nx * ny, MPI_FLOAT, mypid - 1, 0, comm, &mpistatus);

        /* align the first image */
        ccorr2d(nx, ny, imagebuf, imagedata, cccoefs);

        // peak search
        /* peaksearch(nx, ny, cccoefs, relsx, relsy); */
        speak(nx, ny, cccoefs, relsx, relsy);

        if (mypid < ncpus - 1 && mypid < nzsum - 1) {
            // send the last image to the next processor
            MPI_Send(&imagedata[nx * ny * (nzloc - 1)], nx * ny, MPI_FLOAT, mypid + 1, 0, comm);
        }
    } else {
        /* processor idle */
        printf("mypid = %d is idle\n", mypid);
    }
    MPI_Barrier(comm);

    /* accumulate shifts */
    accshifts(comm, nzloc, relsx, abssx);
    accshifts(comm, nzloc, relsy, abssy);

#ifdef DEBUG
    if (mypid == printid || ncpus == 1) {
        for (imgnum = 0; imgnum < nzloc; imgnum++)
            printf("imgnum = %d, sx = %11.3e, sy = %11.3e, sxacc = %11.3e, syacc = %11.3e\n",
                   imgnum, relsx[imgnum], relsy[imgnum], abssx[imgnum], abssy[imgnum]);
    }
#endif

    /* crop images using Dual & Christian's trick,
       hard code ibeg, iend, set lcut, rcut if not specified in the input */

    if (lcut == -1) lcut = nx / 4 - 1;
    if (rcut == -1) rcut = nx * 3 / 4;

    prj1d = (float *)malloc(nx * sizeof(float));
#ifdef DEBUG
    printf("lcut = %d, rcut = %d\n", lcut, rcut);
#endif

    edgeleft = (int *)malloc(nzloc * sizeof(int));
    edgeright = (int *)malloc(nzloc * sizeof(int));

    /* for MRC format cylinder lays horizontally
       column indices are the fastest changing indices
       for each image average the columns to get a smoother 1-d profile
       cut off parts of the 1-d profile that is larger that EDGETHRESH
       look for the first and last local minima */

    for (imgnum = 0; imgnum < nzloc; imgnum++) {
        for (i = 0; i < nx; i++)
            prj1d[i] = 0.0;
        for (j = 0; j < ny; j++) {
            for (i = 0; i < nx; i++)
                prj1d[i] = prj1d[i] + imagedata(i, j, imgnum);
        }
        for (i = 0; i < nx; i++) {
            prj1d[i] = prj1d[i] / (float)ny;
            if (prj1d[i] > EDGETHRESH) prj1d[i] = EDGETHRESH;
        }

        for (i = 0; i < nx - 3; i++) {
            if ((prj1d[i] - prj1d[i + 1]) * (prj1d[i + 1] * prj1d[i + 2]) < 0) {
                edgeleft[imgnum] = i;
                break;
            }
        }
        if (i == nx - 3) printf("failed to find the left edge\n");
        for (i = nx - 1; i >= 2; i--) {
            if ((prj1d[i] - prj1d[i - 1]) * (prj1d[i - 1] * prj1d[i - 2]) < 0) {
                edgeright[imgnum] = i;
                break;
            }
        }
#ifdef DEBUG
        if (mypid == 0)
            printf("i = %d, edgeleft = %d, edgeright = %d, diff = %d\n", imgnum + 1,
                   edgeleft[imgnum] + 1, edgeright[imgnum] + 1,
                   edgeright[imgnum] - edgeleft[imgnum] + 1);
#endif

        if (i == 1) {
            printf("failed to find the right edge, mypid = %d, imgnum = %d\n", mypid, imgnum);
            MPI_Abort(comm, ierr);
        }
    }

    /* set a fudgefactor if not specified in the input */
    if (fudgefactor == -1) fudgefactor = 40;
    tubewidthloc = nx;
    for (imgnum = 0; imgnum < nzloc; imgnum++)
        if (edgeright[imgnum] - edgeleft[imgnum] + 1 < tubewidthloc)
            tubewidthloc = edgeright[imgnum] - edgeleft[imgnum] + 1;

#ifdef DEBUG
    if (mypid == 0) printf("tubewidthloc = %d\n", tubewidthloc);
#endif

    MPI_Allreduce(&tubewidthloc, &tubewidth, 1, MPI_INT, MPI_MIN, comm);
    tubewidth = tubewidth - fudgefactor;
#ifdef DEBUG
    if (mypid == 0) printf("tubewidth = %d\n", tubewidth);
#endif
    if (tubewidth <= 0) goto EXIT;

    /* create an cropped image */
    arecAllocateCBImage(&croppedimages, tubewidth, ny, nzloc);
    croppeddata = croppedimages.data;

    aligneddata = alignedstack.data;

    for (imgnum = 0; imgnum < nzloc; imgnum++) {
        slack = edgeright[imgnum] - edgeleft[imgnum] - tubewidth;
        ibeg = edgeleft[imgnum] + floor(slack / 2);
        iend = ibeg + tubewidth;
        for (j = 0; j < ny; j++)
            for (i = ibeg; i < iend; i++)
                croppeddata(i - ibeg, j, imgnum) = imagedata(i, j, imgnum);
    }

    /* croppedimages.gatherwrite("croppedimages.mrc",comm); */

    /* thresholding all images in the cropped stack */
    for (imgnum = 0; imgnum < nzloc; imgnum++) {
        for (j = 0; j < ny; j++)
            for (i = 0; i < tubewidth; i++)
                if (croppeddata(i, j, imgnum) > CROPTHRESH) croppeddata(i, j, imgnum) = CROPTHRESH;
    }

    /* align the cropped images once again */
    for (imgnum = 1; imgnum < nzloc; imgnum++) {
        ccorr2d(tubewidth, ny, &croppeddata[tubewidth * ny * (imgnum - 1)],
                &croppeddata[tubewidth * ny * imgnum], cccoefs);

        /* peaksearch(tubewidth, ny, cccoefs,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           &relsx_crop[imgnum],
           &relsy_crop[imgnum]); */
        speak(tubewidth, ny, cccoefs, &relsx_crop[imgnum], &relsy_crop[imgnum]);
    }

    /* take care of the images on the stack boundary*/
    if (mypid == 0) {
        relsx_crop[0] = 0.0;
        relsy_crop[0] = 0.0;
        if (ncpus > 1) {
            /* send the last image to the next processor */
            MPI_Send(&croppeddata[tubewidth * ny * (nzloc - 1)], tubewidth * ny, MPI_FLOAT, 1, 0,
                     comm);
        }
    } else if (mypid < nzsum) {
        /* receive an image to the previous processor to align
           the first image */

        MPI_Recv(imagebuf, tubewidth * ny, MPI_FLOAT, mypid - 1, 0, comm, &mpistatus);

        /* align the first image */
        ccorr2d(tubewidth, ny, imagebuf, croppeddata, cccoefs);

        /* peak search */
        /* peaksearch(tubewidth, ny, cccoefs, relsx_crop, relsy_crop); */
        speak(tubewidth, ny, cccoefs, relsx_crop, relsy_crop);

        if (mypid < ncpus - 1 && mypid < nzsum - 1) {
            /* send the last image to the next processor */
            MPI_Send(&croppeddata[tubewidth * ny * (nzloc - 1)], tubewidth * ny, MPI_FLOAT,
                     mypid + 1, 0, comm);
        }
    } else {
        /* processor idle */
        printf("mypid = %d is idle\n", mypid);
    }

    MPI_Barrier(comm);

    accshifts(comm, nzloc, relsx_crop, abssx_crop);
    accshifts(comm, nzloc, relsy_crop, abssy_crop);

    if (mypid == printid || ncpus == 1) {
#ifdef DEBUG
        for (imgnum = 0; imgnum < nzloc; imgnum++)
            printf("imgnum = %d, sx = %11.3e, sy = %11.3e, sxacc = %11.3e, syacc = %11.3e\n",
                   imgnum, relsx_crop[imgnum], relsy_crop[imgnum], abssx_crop[imgnum],
                   abssy_crop[imgnum]);
#endif
    }

    /* global translation alignment using the 0 and 180 degree images */
    /* first figure where the image associated 180 degree is */
    psize = (int *)calloc(ncpus, sizeof(int));
    nbase = (int *)calloc(ncpus, sizeof(int));
    for (i = 0; i < ncpus; i++) {
        psize[i] = nzsum / ncpus;
        if (i < nzsum % ncpus) psize[i] = psize[i] + 1;
    }
    nbase[0] = 0;
    sid = 0;
    iloc = 0;
    for (i = 1; i < ncpus; i++) {
        nbase[i] = nbase[i - 1] + psize[i - 1];
    }
    for (i = 0; i < ncpus; i++) {
        if (nbase[i] + psize[i] > iflip) {
            sid = i;
            iloc = iflip - nbase[i];
            if (mypid == 0) printf("sid = %d, iloc = %d\n", sid, iloc);
            break;
        }
    }
    free(psize);
    free(nbase);

    if (ncpus > 1) {
        if (mypid == 0) { MPI_Recv(imagebuf, nx * ny, MPI_FLOAT, sid, 0, comm, &mpistatus); }
        if (mypid == sid) {
            float *mirror_image;
            mirror_image = (float *)malloc(nx * ny * sizeof(float));
            /* shift image first using previously determined shifts */
            if (nzloc > 0) {
                /* icshift2d(nx, ny, &imagedata[nx*ny*iloc], mirror_image,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  -abssx[iloc],
                   -abssy_crop[iloc]); */
                fshift2d(nx, ny, &imagedata[nx * ny * iloc], mirror_image, -abssx[iloc],
                         -abssy_crop[iloc]);
            } else {
                printf("Each processor must have at least one image!\n");
                MPI_Abort(comm, ierr);
            }
            MPI_Send(mirror_image, nx * ny, MPI_FLOAT, 0, 0, comm);
            free(mirror_image);
        }
        if (mypid == 0) {
            /* flip in x (horizontal) direction.
               the image is ny (vertical) by nx (horizontal)
               x is the fastest changing direction */
            for (j = 0; j < ny; j++)
                for (i = 0; i < nx; i++)
                    imageflip[nx * j + i] = imagebuf[nx * j + (nx - i)];
        }
    } else {
        /* shift image first using previously determined shifts */
        if (nzloc >= 90) {
            /* icshift2d(nx, ny, &imagedata[nx*ny*90], imagebuf,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              -abssx[90],
               -abssy_crop[90]); */
            fshift2d(nx, ny, &imagedata[nx * ny * 90], imagebuf, -abssx[90], -abssy_crop[90]);
        } else {
            fprintf(stderr, "the input stack should have at least 90 images ");
            fprintf(stderr, "separated by 2 degrees. nzloc = %d", nzloc);
            MPI_Abort(comm, ierr);
        }

        /* flip in x (horizontal) direction.
           the image is ny (vertical) by nx (horizontal)
           x is the fastest changing direction */

        for (j = 0; j < ny; j++)
            for (i = 0; i < nx; i++)
                imageflip[nx * j + i] = imagebuf[nx * j + (nx - i)];
    }

    if (mypid == 0) {
        ccorr2d(nx, ny, imagedata, imageflip, cccoefs);
        /* peak search */
        /* peaksearch(nx, ny, cccoefs, &sx, &sy); */
        speak(nx, ny, cccoefs, &sx, &sy);
        printf("sx = %11.4e, sy = %11.4e\n", sx, sy);
    }

    /* broadcast global sx to all processors to center images */
    MPI_Bcast(&sx, 1, MPI_FLOAT, 0, comm);

    /* shift images based on the cumulative shifts deduced in all previous
       steps */
    for (imgnum = 0; imgnum < nzloc; imgnum++) {
        abssx[imgnum] = abssx[imgnum] - sx / 2.0;
        /* icshift2d(nx, ny, &imagedata[nx*ny*imgnum], &aligneddata[nx*ny*imgnum],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          -abssx[imgnum],
           -abssy_crop[imgnum]); */
        fshift2d(nx, ny, &imagedata[nx * ny * imgnum], &aligneddata[nx * ny * imgnum],
                 -abssx[imgnum], -abssy_crop[imgnum]);
    }
    /* alignedimages.gatherwrite("alignedstack.mrc",comm); */

    MPI_Barrier(comm);

    if (mypid == 0) { printf("Done with align2dstack\n"); }

    /* write shifts to a LOG file */
    if (mypid == 0) {
        fp = fopen("arec3dLOG.iter0", "wb");
        sxbuf = (float *)malloc(nzloc * sizeof(float));
        sybuf = (float *)malloc(nzloc * sizeof(float));
        for (i = 0; i < ncpus; i++) {
            if (i == 0) {
                nimgs = nzloc;
                for (j = 0; j < nimgs; j++) {
                    sxbuf[j] = abssx[j];
                    sybuf[j] = abssy_crop[j];
                }
            } else {
                /* Receive from proc i */
                MPI_Recv(&nimgs, 1, MPI_INT, i, 0, comm, &mpistatus);
                MPI_Recv(sxbuf, nimgs, MPI_FLOAT, i, 1, comm, &mpistatus);
                MPI_Recv(sybuf, nimgs, MPI_FLOAT, i, 2, comm, &mpistatus);
            }
            for (j = 0; j < nimgs; j++)
                fprintf(fp, "%11.4e  %11.4e  %11.4e\n", sxbuf[j], sybuf[j], 0.0);
        }
        fclose(fp);
        free(sxbuf);
        free(sybuf);
    } else {
        MPI_Send(&nzloc, 1, MPI_INT, 0, 0, comm);
        MPI_Send(abssx, nzloc, MPI_INT, 0, 1, comm);
        MPI_Send(abssy_crop, nzloc, MPI_INT, 0, 2, comm);
    }
    MPI_Barrier(comm);

EXIT:
    fftwf_free(cccoefs);
    fftwf_free(relsx);
    fftwf_free(relsy);
    fftwf_free(abssx);
    fftwf_free(abssy);
    fftwf_free(abssx_crop);
    fftwf_free(abssy_crop);
    fftwf_free(relsx_crop);
    fftwf_free(relsy_crop);
    fftwf_free(imagebuf);
    fftwf_free(imageflip);
    fftwf_free(prj1d);
    free(edgeleft);
    free(edgeright);
}

/* perform2d circular (aliased) cross-correlation
   nx is associated with the fastest growing index */
void ccorr1d(int nx, float *x, float *y, float *c) {
    /* basic algorithm:
       fx <--- fft(x);
       fy <--- fft(y);
       ff = conj(fx).*fy;
       c <--- ifft(ff);
    */

    int i;
    fftwf_plan px, py, pc;

    fftwf_complex *fx = (fftwf_complex *)fftwf_malloc((nx / 2 + 1) * sizeof(fftwf_complex));
    fftwf_complex *fy = (fftwf_complex *)fftwf_malloc((nx / 2 + 1) * sizeof(fftwf_complex));
    fftwf_complex *ff = (fftwf_complex *)fftwf_malloc((nx / 2 + 1) * sizeof(fftwf_complex));

    /* FFT the image x */
    px = fftwf_plan_dft_r2c_1d(nx, x, fx, FFTW_ESTIMATE);
    fftwf_execute(px);

    /* FFT the image y */
    py = fftwf_plan_dft_r2c_1d(nx, y, fy, FFTW_ESTIMATE);
    fftwf_execute(py);

    /* conj(fx).*fy */
    for (i = 0; i < nx / 2 + 1; i++) {
        ff[i][0] = fx[i][0] * fy[i][0] + fx[i][1] * fy[i][1];
        ff[i][1] = fx[i][0] * fy[i][1] - fx[i][1] * fy[i][0];
    }

    pc = fftwf_plan_dft_c2r_1d(nx, ff, c, FFTW_ESTIMATE);
    fftwf_execute(pc);

    fftwf_destroy_plan(px);
    fftwf_destroy_plan(py);
    fftwf_destroy_plan(pc);

    fftwf_free(fx);
    fftwf_free(fy);
    fftwf_free(ff);
}

/*----------------------------------------------------------*/

void ccorr2d(int nx, int ny, float *x, float *y, float *c) {
    /* basic algorithm:
       fx <--- fft2d(x);
       fy <--- fft2d(y);
       ff = conj(fx).*fy;
       c <--- ifft2d(ff);
    */
    int i;
    fftwf_plan px, py, pc;

    fftwf_complex *fx = (fftwf_complex *)fftwf_malloc(ny * (nx / 2 + 1) * sizeof(fftwf_complex));
    fftwf_complex *fy = (fftwf_complex *)fftwf_malloc(ny * (nx / 2 + 1) * sizeof(fftwf_complex));
    fftwf_complex *ff = (fftwf_complex *)fftwf_malloc(ny * (nx / 2 + 1) * sizeof(fftwf_complex));

    /* FFT the image x */
    px = fftwf_plan_dft_r2c_2d(ny, nx, x, fx, FFTW_ESTIMATE);
    fftwf_execute(px);

    /* FFT the image y */
    py = fftwf_plan_dft_r2c_2d(ny, nx, y, fy, FFTW_ESTIMATE);
    fftwf_execute(py);

    /* conj(fx).*fy */
    for (i = 0; i < ny * (nx / 2 + 1); i++) {
        ff[i][0] = fx[i][0] * fy[i][0] + fx[i][1] * fy[i][1];
        ff[i][1] = fx[i][0] * fy[i][1] - fx[i][1] * fy[i][0];
        // float temp=sqrt(ff[i][0]*ff[i][0]+ff[i][1]*ff[i][1]);
        // ff[i][0]/=temp;
        // ff[i][1]/=temp;
    }

    pc = fftwf_plan_dft_c2r_2d(ny, nx, ff, c, FFTW_ESTIMATE);
    fftwf_execute(pc);

    fftwf_destroy_plan(px);
    fftwf_destroy_plan(py);
    fftwf_destroy_plan(pc);

    fftwf_free(fx);
    fftwf_free(fy);
    fftwf_free(ff);
}

/*--------------------------------------------------------*/
#define cccoefs(i, j) cccoefs[(j)*nx + (i)]
void peaksearch(int nx, int ny, float *cccoefs, int *ix, int *iy) {
    int i, j;
    float pmax = 0.0;
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
            if (fabs(cccoefs(i, j)) > pmax) {
                *ix = i;
                *iy = j;
                pmax = cccoefs(i, j);
            }
        }

    if (*ix > nx / 2) *ix = *ix - nx;
    if (*iy > nx / 2) *iy = *iy - ny;
}

#undef cccoefs

/*--------------------------------------------------------*/
#define imagein(i, j) imagein[(j)*nx + (i)]
#define imageout(i, j) imageout[(j)*nx + (i)]

void icshift2d(int nx, int ny, float *imagein, float *imageout, int ix, int iy) {
    int sx, sy, is, js, j, i;

    // make sure shifts are positive
    sx = ix % nx;
    sy = iy % ny;
    if (sx < 0) sx = sx + nx;
    if (sy < 0) sy = sy + ny;

    // circular shift
    for (j = 0; j < ny; j++) {
        js = (j + sy) % ny;
        for (i = 0; i < nx; i++) {
            is = (i + sx) % nx;
            imageout(is, js) = imagein(i, j);
        }
    }
}
/*--------------------------------------------------------*/
void fshift2d(int nx, int ny, float *imagein, float *imageout, float sx, float sy) {
    /* shift image by applying a phase factor to the Fourier transform of
       the image */
    fftwf_plan px, pc;
    float phase, cosph, sinph, fr, fi;
    int i, j;
    float *kx, *ky;

    fftwf_complex *cx, *ff;

    /* put imagein in a complex array */
    cx = (fftwf_complex *)fftwf_malloc(ny * nx * sizeof(fftwf_complex));
    for (i = 0; i < nx * ny; i++) {
        cx[i][0] = imagein[i];
        cx[i][1] = 0.0;
    }

    /* FFT imagein */
    ff = (fftwf_complex *)fftwf_malloc(ny * nx * sizeof(fftwf_complex));
    px = fftwf_plan_dft_2d(ny, nx, cx, ff, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(px);

    /* set up the reciprocal space vectors that preserve conjugate
       symmetry */
    kx = (float *)malloc(nx * sizeof(float));
    ky = (float *)malloc(ny * sizeof(float));

    for (i = 0; i < ceil((float)nx / 2.0); i++)
        kx[i] = (float)i / (float)nx;
    for (i = ceil((float)nx / 2.0); i < nx; i++)
        kx[i] = (float)(i - nx) / (float)(nx);

    for (i = 0; i < ceil((float)ny / 2.0); i++)
        ky[i] = (float)i / (float)ny;
    for (i = ceil((float)ny / 2.0); i < ny; i++)
        ky[i] = (float)(i - ny) / (float)ny;

    /* multiply the phase factor */
    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
            phase = -2.0 * piFunc() * (sx * kx[i] + sy * ky[j]);
            cosph = cos(phase);
            sinph = sin(phase);
            fr = ff[nx * j + i][0];
            fi = ff[nx * j + i][1];
            ff[nx * j + i][0] = fr * cosph - fi * sinph;
            ff[nx * j + i][1] = fr * sinph + fi * cosph;
        }
    }

    pc = fftwf_plan_dft_2d(ny, nx, ff, cx, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(pc);

    for (i = 0; i < nx * ny; i++) {
        imageout[i] = cx[i][0] / (float)(nx * ny);
    }

    fftwf_destroy_plan(px);
    fftwf_destroy_plan(pc);

    fftwf_free(cx);
    fftwf_free(ff);
    free(kx);
    free(ky);
}

#undef imagein
#undef imageout
/*--------------------------------------------------------*/
void iaccshifts(MPI_Comm comm, int nloc, int *shiftin, int *shiftout) {
    int mypid, ncpus, img;
    MPI_Status mpistatus;

    MPI_Comm_size(comm, &ncpus);
    MPI_Comm_rank(comm, &mypid);

    if (mypid == 0) {
        shiftout[0] = 0.0;
    } else {
        MPI_Recv(&shiftout[0], 1, MPI_INT, mypid - 1, 0, comm, &mpistatus);
        shiftout[0] = shiftout[0] + shiftin[0];
    }

    for (img = 1; img < nloc; img++)
        shiftout[img] = shiftout[img - 1] + shiftin[img];

    if (ncpus > 1) {
        if (mypid < ncpus - 1) { MPI_Send(&shiftout[nloc - 1], 1, MPI_INT, mypid + 1, 0, comm); }
    }
}
/*--------------------------------------------------------*/
void accshifts(MPI_Comm comm, int nloc, float *shiftin, float *shiftout) {
    int mypid, ncpus, img;
    MPI_Status mpistatus;

    MPI_Comm_size(comm, &ncpus);
    MPI_Comm_rank(comm, &mypid);

    if (mypid == 0) {
        shiftout[0] = 0.0;
    } else {
        MPI_Recv(&shiftout[0], 1, MPI_INT, mypid - 1, 0, comm, &mpistatus);
        shiftout[0] = shiftout[0] + shiftin[0];
    }

    for (img = 1; img < nloc; img++)
        shiftout[img] = shiftout[img - 1] + shiftin[img];

    if (ncpus > 1) {
        if (mypid < ncpus - 1) { MPI_Send(&shiftout[nloc - 1], 1, MPI_INT, mypid + 1, 0, comm); }
    }
}
/*--------------------------------------------------------*/
/* generate the self-correlation function (SCF, see van Heel et al.
   Ultramicroscopy 46, 1992, pp 307--316) of x and put it in y.
   ny is the row dimension, ny is the column dimension
   nx is associated with the fastest growing index */

void getscf(int nx, int ny, float *x, float *y) {
    /* basic algorithm:
       fx <--- fft2d(x);
       fy <--- abs(fx);
       y  <--- ifft2d(fy);
    */
    int i;

    fftwf_complex *fx = (fftwf_complex *)fftwf_malloc(ny * (nx / 2 + 1) * sizeof(fftwf_complex));
    fftwf_complex *fy = (fftwf_complex *)fftwf_malloc(ny * (nx / 2 + 1) * sizeof(fftwf_complex));

    fftwf_plan px, py;

    /* FFT the image x */
    px = fftwf_plan_dft_r2c_2d(ny, nx, x, fx, FFTW_ESTIMATE);
    fftwf_execute(px);

    /* fy = abs(fx) */
    for (i = 0; i < ny * (nx / 2 + 1); i++) {
        /* if sqrt is not used, then autocorrelation (acf) is produced */
        fy[i][0] = sqrt(fx[i][0] * fx[i][0] + fx[i][1] * fx[i][1]);
        fy[i][1] = 0.0;
    }

    py = fftwf_plan_dft_c2r_2d(ny, nx, fy, y, FFTW_ESTIMATE);
    fftwf_execute(py);

    fftwf_destroy_plan(px);
    fftwf_destroy_plan(py);

    fftwf_free(fx);
    fftwf_free(fy);
}

    /* change from image stored in Cartesian coordinates to that stored in
       polar coordinates using linear interpolation */

#define cdata(i, j) cdata[nx * (j) + (i)]
#define pdata(i, j) pdata[nang * (j) + (i)]

/*  transform from Cartesian to polar coordinates using linear interpolation */
void cart2po(int nx, int ny, float *cdata, int nang, int r1, int r2, float *pdata) {
    int i, j;
    float ang, x, y, xcent, ycent, dx, dy, dmx, dmy;
    int xf, yf, xc, yc;

    xcent = nx / 2 + 1;
    ycent = ny / 2 + 1;
    if (r2 > nx / 2 || r2 > ny / 2) {
        fprintf(stderr, "r2 is too large: r2 = %d\n", r2);
        exit(1);
    }

    for (j = r1; j <= r2; j++) {
        for (i = 0; i < nang; i++) {

            /* Compute the angle. */
            ang = 2.0 * piFunc() / ((float)nang * i);

            x = j * cos(ang) + xcent;
            y = j * sin(ang) + ycent;

            /* Linear interpolation. */
            xf = floor(x);
            xc = ceil(x);
            yf = floor(y);
            yc = ceil(y);

            dx = x - xf;
            dmx = 1.0 - dx;
            dy = y - yf;
            dmy = 1.0 - dy;
            if (xf >= 0 && xf <= nx - 1 && yf >= 0 && yf <= ny - 1) {
                pdata(i, j - r1) =
                    sqrt((float)j) * (cdata(xf, yf) * dmx * dmy + cdata(xf, yc) * dmx * dy +
                                      cdata(xc, yf) * dx * dmy + cdata(xc, yc) * dx * dy);
            } else {
                printf("Out of bound. i,j = %d,%d\n", i, j);
                printf("x = %f, y = %f, xf = %d, yf = %d\n", x, y, xf, yf);
                exit(1);
                pdata(i, j - r2) = 0.0;
            } /* end if */
        }     /* end for i */
    }         /* end for j */
}
#undef cdata
#undef pdata

#define imageout(i, k) imageout[NSAM * ((i)-1) + (k)-1]
#define imagein(i) imagein[(i)-1]
void rotate2d(int NSAM, int NROW, float *imagein, float *imageout, float THETA) {
    /* both imagein and imageout are assumed to be NSAM by NROW
       in column major order, i.e., NSAM is the leading dimension */

    int NSAMH, NROWH, KCENT, ICENT, JJ;
    double YCOD, YSID, COD, SID, X, Y, XOLD, YOLD, YDIF, YREM, XDIF, XREM;
    int I, K, IXOLD, IYOLD, NADDR;
    double PI = piFunc();

    THETA = THETA * piFunc() / 180.0;

    NSAMH = NSAM / 2;
    NROWH = NROW / 2;

    KCENT = NSAMH + 1;
    ICENT = NROWH + 1;

    if (THETA > PI) THETA = -2.0 * PI + THETA;
    if (THETA < -PI) THETA = 2. * PI + THETA;
    COD = cos(THETA);
    SID = sin(THETA);

    JJ = 0;
    for (I = 1; I <= NROW; I++) {
        JJ = JJ + 1;
        Y = I - ICENT;
        YCOD = Y * COD + ICENT;
        YSID = -Y * SID + KCENT;
        for (K = 1; K <= NSAM; K++) {
            imageout(I, K) = 0.0;
            X = K - KCENT;
            XOLD = X * COD + YSID;
            YOLD = X * SID + YCOD;
            IYOLD = YOLD;
            YDIF = YOLD - IYOLD;
            YREM = 1.0 - YDIF;
            IXOLD = XOLD;
            if ((IYOLD >= 1 && IYOLD <= NROW - 1) && (IXOLD >= 1 && IXOLD <= NSAM - 1)) {
                /* INSIDE BOUNDARIES OF OUTPUT IMAGE */
                XDIF = XOLD - IXOLD;
                XREM = 1.0 - XDIF;
                NADDR = (IYOLD - 1) * NSAM + IXOLD;
                imageout(I, K) =
                    YDIF * (imagein(NADDR + NSAM) * XREM + imagein(NADDR + NSAM + 1) * XDIF) +
                    YREM * (imagein(NADDR) * XREM + imagein(NADDR + 1) * XDIF);
            } /* end if */
        }     /* end for K */
    }         /* end for I*/
}
#undef imageout
#undef imagein

/* change from Cartesian to Polar coordinates using quadratic interpolation */
#define Y(i, j) Y[NSAMP * ((j)-1) + (i)-1]
void to_polar(int NSAM, int NROW, int R1, int R2, float *X, float *Y) {
    /* X is of size NROW by NSAM

       Y must be of size (NROWP,NSAMP), where NSAMP = int(2*PI*R2),
       and NROWP = R2-R1+1 */

    int IXC, IYC, NSAMP, NROWP, I, J;
    double DFI, FI, XS, YS;
    double PI = piFunc();

    NSAMP = (int)(2 * piFunc() * R2);
    NROWP = R2 - R1 + 1;

    IXC = NSAM / 2 + 1;
    IYC = NROW / 2 + 1;
    DFI = 2 * PI / NSAMP;

    for (J = R1; J <= R2; J++) {
        for (I = 1; I <= NSAMP; I++) {
            FI = (I - 1) * DFI;
            XS = cos(FI) * J;
            YS = sin(FI) * J;
            Y(I, J - R1 + 1) = QUADRI(XS + IXC, YS + IYC, NSAM, NROW, X) * sqrt((double)J);
        }
    }
}
#undef Y

/*-------------------------------------------------*/
/*
C  FUNCTION QUADRI(XX, YY, NXDATA, NYDATA, FDATA)
C                                                                      *
C  PURPOSE: QUADRATIC INTERPOLATION                                                           *
C                                                                      *
C  PARAMETERS:       XX,YY TREATED AS CIRCULARLY CLOSED.
C                    FDATA - IMAGE 1..NXDATA, 1..NYDATA
C
C                    F3    FC       F0, F1, F2, F3 are the values
C                     +             at the grid points.  X is the
C                     + X           point at which the function
C              F2++++F0++++F1       is to be estimated. (It need
C                     +             not be in the First quadrant).
C                     +             FC - the outer corner point
C                    F4             nearest X.
C
C                                   F0 is the value of the FDATA at
C                                   FDATA(I,J), it is the interior mesh
C                                   point nearest  X.
C                                   The coordinates of F0 are (X0,Y0),
C                                   The coordinates of F1 are (XB,Y0),
C                                   The coordinates of F2 are (XA,Y0),
C                                   The coordinates of F3 are (X0,YB),
C                                   The coordinates of F4 are (X0,YA),
C                                   The coordinates of FC are (XC,YC),
C
C                   O               HXA, HXB are the mesh spacings
C                   +               in the X-direction to the left
C                  HYB              and right of the center point.
C                   +
C            ++HXA++O++HXB++O       HYB, HYA are the mesh spacings
C                   +               in the Y-direction.
C                  HYA
C                   +               HXC equals either  HXB  or  HXA
C                   O               depending on where the corner
C                                   point is located.
c
C                                   Construct the interpolant
C                                   F = F0 + C1*(X-X0) +
C                                       C2*(X-X0)*(X-X1) +
C                                       C3*(Y-Y0) + C4*(Y-Y0)*(Y-Y1)
C                                       + C5*(X-X0)*(Y-Y0)
C
C23456789012345678901234567890123456789012345678901234567890123456789012
*/
/*
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  void QUADRI_FAST(X, Y, NXDATA,
NYDATA, FDATA)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  REAL    :: X,Y
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  REAL    :: FDATA(NXDATA,NYDATA)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  INTEGER :: NXDATA, NYDATA

C     SKIP CIRCULAR CLOSURE, IT IS SLOW, ENSURE IT NOT NEEDED IN CALLER

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  I   = IFIX(X)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  J   = IFIX(Y)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  DX0 = X - I
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  DY0 = Y - J

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  IP1 = I + 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  IM1 = I - 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  JP1 = J + 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  JM1 = J - 1

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  F0  = FDATA(I,J)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  C1  = FDATA(IP1,J) - F0 ! DIFF.
FROM CENTER C2  = (C1 - F0 + FDATA(IM1,J)) * 0.5  ! DIFF OF X+1 AND X-1 C3  = FDATA(I,JP1) - F0 !
DIFF. FROM CENTER C4  = (C3 - F0 + FDATA(I,JM1)) * 0.5  ! DIFF oF Y+1 AND Y-1

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  DXB = (DX0 - 1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  DYB = (DY0 - 1)

C     HXC & HYC ARE EITHER 1 OR -1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  HXC = INT(SIGN(1.0,DX0))   ! X <>
INT(X) HYC = INT(SIGN(1.0,DY0))   ! Y <> INT(Y)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  IC  = I + HXC
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  JC  = J + HYC

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  C5  =  ((FDATA(IC,JC) - F0 -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 &         HXC * C1 -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 &        (HXC * (HXC - 1.0)) * C2 -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 &         HYC * C3 -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 &        (HYC * (HYC - 1.0)) * C4)
* &        (HXC * HYC))

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  QUADRI_FAST = F0 +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 &         DX0 * (C1 + DXB * C2 +
DY0 * C5) + &         DY0 * (C3 + DYB * C4)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  END
*/
/*
C     ------------------- QUADRI -----------------------------------
*/
#define FDATA(i, j) FDATA[NXDATA * ((i)-1) + (j)-1]
#define SIGN(a) ((a) >= 0.0 ? 1 : -1)

double QUADRI(double XX, double YY, int NXDATA, int NYDATA, float *FDATA) {
    /* DIMENSION  FDATA(NXDATA,NYDATA) */
    double X, Y, DX0, DY0, F0, C1, C2, C3, C4, C5, DXB, DYB, QUADRI;
    int I, J, IP1, IM1, JP1, JM1, HXC, HYC, IC, JC;

    X = XX;
    Y = YY;

    /*  CIRCULAR CLOSURE */
    if (X < 1.0) X = X + (1 - (int)(X) / NXDATA) * NXDATA;
    if (X > (double)(NXDATA) + 0.5) X = fmod(X - 1.0, (double)(NXDATA)) + 1.0;
    if (Y < 1.0) Y = Y + (1 - (int)(Y) / NYDATA) * NYDATA;
    if (Y > (double)(NYDATA) + 0.5) Y = fmod(Y - 1.0, (double)(NYDATA)) + 1.0;

    I = (int)(X);
    J = (int)(Y);

    DX0 = X - I;
    DY0 = Y - J;

    IP1 = I + 1;
    IM1 = I - 1;
    JP1 = J + 1;
    JM1 = J - 1;

    if (IP1 > NXDATA) IP1 = IP1 - NXDATA;
    if (IM1 < 1) IM1 = IM1 + NXDATA;
    if (JP1 > NYDATA) JP1 = JP1 - NYDATA;
    if (JM1 < 1) JM1 = JM1 + NYDATA;

    F0 = FDATA(I, J);
    C1 = FDATA(IP1, J) - F0;
    C2 = (C1 - F0 + FDATA(IM1, J)) * 0.5;
    C3 = FDATA(I, JP1) - F0;
    C4 = (C3 - F0 + FDATA(I, JM1)) * 0.5;

    DXB = DX0 - 1;
    DYB = DY0 - 1;

    /*  HXC & HYC ARE EITHER 1 OR -1 */
    HXC = SIGN(DX0);
    HYC = SIGN(DY0);

    IC = I + HXC;
    JC = J + HYC;

    if (IC > NXDATA) {
        IC = IC - NXDATA;
    } else if (IC < 1) {
        IC = IC + NXDATA;
    }

    if (JC > NYDATA) {
        JC = JC - NYDATA;
    } else if (JC < 1) {
        JC = JC + NYDATA;
    }

    C5 = ((FDATA(IC, JC) - F0 - HXC * C1 - (HXC * (HXC - 1.0)) * C2 - HYC * C3 -
           (HYC * (HYC - 1.0)) * C4) *
          (HXC * HYC));

    QUADRI = F0 + DX0 * (C1 + DXB * C2 + DY0 * C5) + DY0 * (C3 + DYB * C4);

    return QUADRI;
}
#undef FDATA
/*---------------------------------------------*/

/* fit points with a 1D polynomial */
#define B(i) B[(i)-1]
void PRB1D(double *B, int NPOINT, double *POS) {
    /* array B(NPOINT) */
    double C2, C3;
    int NHALF;

    NHALF = NPOINT / 2 + 1;
    *POS = 0.0;
    if (NPOINT == 7) {
        C2 = 49. * B(1) + 6. * B(2) - 21. * B(3) - 32. * B(4) - 27. * B(5) - 6. * B(6) + 31. * B(7);
        C3 = 5. * B(1) - 3. * B(3) - 4. * B(4) - 3. * B(5) + 5. * B(7);
    } else if (NPOINT == 5) {
        C2 = (74. * B(1) - 23. * B(2) - 60. * B(3) - 37. * B(4) + 46. * B(5)) / (-70.);
        C3 = (2. * B(1) - B(2) - 2. * B(3) - B(4) + 2. * B(5)) / 14.;
    } else if (NPOINT == 3) {
        C2 = (5. * B(1) - 8. * B(2) + 3. * B(3)) / (-2.);
        C3 = (B(1) - 2. * B(2) + B(3)) / 2.;
    } else if (NPOINT == 9) {
        C2 = (1708. * B(1) + 581. * B(2) - 246. * B(3) - 773. * B(4) - 1000. * B(5) - 927. * B(6) -
              554. * B(7) + 119. * B(8) + 1092. * B(9)) /
             (-4620.);
        C3 = (28. * B(1) + 7. * B(2) - 8. * B(3) - 17. * B(4) - 20. * B(5) - 17. * B(6) -
              8. * B(7) + 7. * B(8) + 28. * B(9)) /
             924.0;
    }

    if (C3 != 0.0) *POS = C2 / (2. * C3) - NHALF;
}
#undef B
/*---------------------------------------------*/
#define rsq(i, j) rsq[((j) + 1) * 3 + (i) + 1]
#define data(i, j) data[((j)-1) * nx + (i)-1]
void speak(int nx, int ny, float *data, float *xt, float *yt) {
    /*
       data is an nx by ny array
    C
    C  PURPOSE:  SEARCHES FOR THE ML HIGHEST PEAKS IN THE (REAL) IMAGE
    C            FILNAM AND PRINTS OUT POSITIONS AND VALUES OF THESE PEAKS.
    C
    */
    int i, j, ix, iy, irow, jcol;
    float rsq[9];
    float xsh, ysh, peakv;
    float pmax = 0.0;

    for (i = 1; i <= nx; i++)
        for (j = 1; j <= ny; j++) {
            if (fabs(data(i, j)) > pmax) {
                ix = i;
                iy = j;
                pmax = data(i, j);
            }
        }

    /* put the 3x3 pixels around the peak (ix,iy) into rsq */
    for (j = -1; j <= 1; j++) {
        jcol = iy + j;
        if (jcol < 1) jcol = jcol + ny;
        if (jcol > ny) jcol = jcol - ny;
        for (i = -1; i <= 1; i++) {
            irow = ix + i;
            if (irow < 1) irow = irow + nx;
            if (irow > nx) irow = irow - nx;
            rsq(i, j) = data(irow, jcol);
        } /* end for j */
    }     /* end for i */

    quad2dfit(rsq, &xsh, &ysh, &peakv);

    /* change to zero based index */
    ix--;
    iy--;

    *xt = xsh + ix;
    *yt = ysh + iy;

    if (*xt > nx / 2) *xt = *xt - nx;
    if (*yt > ny / 2) *yt = *yt - ny;
}
#undef rsq
#undef data
    /*---------------------------------------------*/
    /* quad2dfit
    C
    C PARABOLIC FIT TO 3 BY 3 PEAK NEIGHBORHOOD
    C
    C THE FORMULA FOR PARABOLOID TO BE FIITED INTO THE NINE POINTS IS:
    C
    C	F = C1 + C2*Y + C3*Y**2 + C4*X + C5*XY + C6*X**2
    C
    C THE VALUES OF THE COEFFICIENTS C1 - C6 ON THE BASIS OF THE
    C NINE POINTS AROUND THE PEAK, AS EVALUATED BY ALTRAN:
    C
    C adapted from fortran code in SPIDER
    */

#define Z(i, j) Z[((j)-1) * 3 + (i)-1]

void quad2dfit(float *Z, float *XSH, float *YSH, float *PEAKV) {
    double C1, C2, C3, C4, C5, C6, DENOM;

    C1 = (26.0 * Z(1, 1) - Z(1, 2) + 2.0 * Z(1, 3) - Z(2, 1) - 19.0 * Z(2, 2) - 7.0 * Z(2, 3) +
          2.0 * Z(3, 1) - 7.0 * Z(3, 2) + 14.0 * Z(3, 3)) /
         9.0;

    C2 = (8. * Z(1, 1) - 8. * Z(1, 2) + 5. * Z(2, 1) - 8. * Z(2, 2) + 3. * Z(2, 3) + 2. * Z(3, 1) -
          8. * Z(3, 2) + 6. * Z(3, 3)) /
         (-6.);

    C3 = (Z(1, 1) - 2. * Z(1, 2) + Z(1, 3) + Z(2, 1) - 2. * Z(2, 2) + Z(2, 3) + Z(3, 1) -
          2. * Z(3, 2) + Z(3, 3)) /
         6.0;

    C4 = (8. * Z(1, 1) + 5. * Z(1, 2) + 2. * Z(1, 3) - 8. * Z(2, 1) - 8. * Z(2, 2) - 8. * Z(2, 3) +
          3. * Z(3, 2) + 6. * Z(3, 3)) /
         (-6.0);

    C5 = (Z(1, 1) - Z(1, 3) - Z(3, 1) + Z(3, 3)) / 4.0;

    C6 = (Z(1, 1) + Z(1, 2) + Z(1, 3) - 2. * Z(2, 1) - 2. * Z(2, 2) - 2. * Z(2, 3) + Z(3, 1) +
          Z(3, 2) + Z(3, 3)) /
         6.0;

    /* THE PEAK COORDINATES OF THE PARABOLOID CAN NOW BE EVALUATED AS: */

    *YSH = 0.0;
    *XSH = 0.0;
    DENOM = 4. * C3 * C6 - C5 * C5;
    if (DENOM != 0.0) {
        *YSH = (C4 * C5 - 2. * C2 * C6) / DENOM - 2.0;
        *XSH = (C2 * C5 - 2. * C4 * C3) / DENOM - 2.0;
        *PEAKV = 4. * C1 * C3 * C6 - C1 * C5 * C5 - C2 * C2 * C6 + C2 * C4 * C5 - C4 * C4 * C3;
        *PEAKV = (*PEAKV) / DENOM;
        /* LIMIT INTERPLATION TO +/- 1. RANGE */
        if (*YSH < -1.0) *YSH = -1.0;
        if (*YSH > 1.0) *YSH = +1.0;
        if (*XSH < -1.0) *XSH = -1.0;
        if (*XSH > 1.0) *XSH = +1.;
    }
}
#undef Z
