
#include "arecImage.h"
#include "align2dstack.h"
#include "arecConstants.h"
#include "fftw3.h"
#include "imagetools.h"
#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void to_polar(int NSAM, int NROW, int R1, int R2, float *X, float *Y);
void PRB1D(double *B, int NPOINT, double *POS);

int arecAllocateCBImage(arecImage *cbimage, int nx, int ny, int nz) {
    int status = 0;
    cbimage->nx = nx;
    cbimage->ny = ny;
    cbimage->nz = nz;
    cbimage->data = (float *)calloc(nx * ny * nz, sizeof(float));
    if (!cbimage->data) status = -1;

    cbimage->is_cyl = 0;
    cbimage->cord = NULL;

    return status;
}

int arecAllocateCylImage(arecImage *cylimage, double radius, int height) {
    int nrays, nnz, nx, ny, nz;
    int status = 0;
    int ix, iz, zz, xx, zs, xs, r2;
    double xcent, zcent;
    int *cord;

    nx = static_cast<int>(round(2 * radius + 1));
    nz = static_cast<int>(round(2 * radius + 1));
    ny = height;
    cylimage->nx = nx;
    cylimage->ny = ny;
    cylimage->nz = nz;

    cylimage->radius = radius;
    cylimage->height = height;

    r2 = radius * radius;
    xcent = radius;
    zcent = radius;

    nrays = 0;
    nnz = 0;
    for (iz = 0; iz < nz; iz++) {
        zs = iz - zcent;
        zz = zs * zs;
        for (ix = 0; ix < nx; ix++) {
            xs = ix - xcent;
            xx = xs * xs;
            if ((zz + xx) <= r2) nrays++;
        } /* end for ix */
    }     /* end for iz */
    nnz = nrays * height;

    cylimage->nrays = nrays;
    cylimage->nnz = nnz;
    cylimage->data = (float *)calloc(nnz, sizeof(float));
    cylimage->cord = (int *)calloc(nrays * 2, sizeof(int));
    if (!cylimage->data) status = -1;

    cylimage->is_cyl = 1;
    cord = cylimage->cord;
    nnz = 0;
    for (iz = 0; iz < nz; iz++) {
        zs = iz - zcent;
        zz = zs * zs;
        for (ix = 0; ix < nx; ix++) {
            xs = ix - xcent;
            xx = xs * xs;
            if ((zz + xx) <= r2) {
                cord(0, nnz) = ix;
                cord(1, nnz) = iz;
                nnz++;
            }
        } /* end for ix */
    }     /* end for iz */
    return status;
}

int ImageCB2Cyl(arecImage cbimage, arecImage cylimage, int xcent, int zcent, double radius,
                int height) {
    int nx, ny, nz, r2, xs, xx, zz, zs, ix, iy, iz, nnz, nrays, nnz0, nrays0;
    float *cylval, *cube;
    int *cord;
    int status = 0;

    nx = cbimage.nx;
    ny = cbimage.ny;
    nz = cbimage.nz;
    cube = cbimage.data;

    /* assume cylimage has been allocated */
    if (!cylimage.data) {
        fprintf(stderr, "ImageCB2Cyl: cylimage not properly allocated\n");
        status = -1;
        return status;
    }

    radius = cylimage.radius;
    height = cylimage.height;
    nnz0 = cylimage.nnz;
    nrays0 = cylimage.nrays;
    cylval = cylimage.data;
    cord = cylimage.cord;

    xcent = radius;
    zcent = radius;

    if (nnz0 <= 0 || nrays0 <= 0 || radius <= 0 || height <= 0) {
        fprintf(stderr, "ImageCB2Cy: cylimage not properly created!\n");
        fprintf(stderr, "nnz,nrays,radius,height=%d,%d,%d,%d\n", nnz0, nrays0, radius, height);
        status = 1;
        goto EXIT;
    }

    if (!cylval || !cord) {
        fprintf(stderr, "ImageCB2Cyl: cylimage not properly allocated\n");
        status = -1;
        goto EXIT;
    }

    nnz = 0;
    nrays = 0;
    r2 = radius * radius;

    for (iz = 0; iz < nz; iz++) {
        zs = iz - zcent;
        zz = zs * zs;
        for (ix = 0; ix < nx; ix++) {
            xs = ix - xcent;
            xx = xs * xs;
            if ((xx + zz) <= r2) {
                for (iy = 0; iy < ny; iy++) {

                    cylval(nnz) = cube(ix, iy, iz);
                    nnz++;
                }
                cord(0, nrays) = ix;
                cord(1, nrays) = iz;
                nrays++;
            } // end if (xx + yy) <= r2
        }     // end for ix
    }         // end for iy

    if (nnz != nnz0 || nrays != nrays0) {
        fprintf(stderr, "ImageCB2Cyl: inconsistency!\n");
        fprintf(stderr, "nnz, nnz0, nrays, nrays0 = %d, %d, %d, %d\n", nnz, nnz0, nrays, nrays0);
        status = -2;
    }

EXIT:
    return status;
}

int ImageCyl2CB(arecImage cylimage, arecImage cbimage, int nx, int ny, int nz) {
    double radius;
    int height, nrays, nnz;
    int ix, iy, iz, j, nnz0;
    int *cord;
    float *cylval = NULL, *cube = NULL;

    int status = 0;

    radius = cylimage.radius;
    height = cylimage.height;
    cord = cylimage.cord;
    nnz0 = cylimage.nnz;
    nrays = cylimage.nrays;
    cylval = cylimage.data;
    if (!cylval) {
        fprintf(stderr, "ImageCyl2CB: cylimage not properly allocated!\n");
        status = -1;
        goto EXIT;
    }

    if (nx != cbimage.nx || ny != cbimage.ny || nz != cbimage.nz) {
        fprintf(stderr, "ImageCyl2CB: inconsistent dimension\n");
        fprintf(stderr, "nx, ny, nz = %d, %d, %d\n", nx, ny, nz);
        fprintf(stderr, "cbnx, cbny, cbnz = %d, %d, %d\n", cbimage.nx, cbimage.ny, cbimage.nz);
        status = 1;
        goto EXIT;
    }

    if (nx != 2 * radius + 1 || nz != 2 * radius + 1 || ny != height) {
        fprintf(stderr, "ImageCyl2CB: inconsistent dimension\n");
        fprintf(stderr, "nx, ny, nz = %d, %d, %d\n", nx, ny, nz);
        fprintf(stderr, "radius = %d, height = %d\n", radius, height);
        status = 1;
        goto EXIT;
    }

    cube = cbimage.data;
    if (!cube) {
        fprintf(stderr, "ImageCyl2CB: cbimage not properly allocated!\n");
        status = -1;
        goto EXIT;
    }

    nnz = 0;
    for (j = 0; j < nrays; j++) {
        ix = cord(0, j);
        iz = cord(1, j);
        if (iz < nz && ix < nx) {
            for (iy = 0; iy < ny; iy++) {

                if (nnz > nnz0) {
                    printf("ImageCyl2CB: iz out of bound, nnz = %d, nnz0 = %d\n", nnz, nnz0);
                    status = -2;
                    goto EXIT;
                }
                cube(ix, iy, iz) = cylval(nnz);
                nnz++;
            }
        } else {
            printf("ImageCyl2CB: ix, iz, out of bound\n");
            printf("ImageCyl2CB: ix = %d, iz = %d \n", ix, iz);
            status = -2;
            goto EXIT;
        }
    }

EXIT:
    return status;
}

void arecImageFree(arecImage *image) {
    if (image->is_cyl) {
        free(image->cord);
        image->cord = NULL;
    }
    free(image->data);
    image->data = NULL;
}

#define fromdata(i, j, k) fromdata[(k)*nx * ny + (j)*nx + (i)]
#define todata(i, j, k) todata[(k)*ldy * ldx + (j)*ldx + (i)]

void arecCropImages(arecImage images, arecImage croppedimages, int xcent, int ycent) {
    int height, length, ldx, ldy, i, j, k, ri, hi;
    int nx, ny, nimages;
    float *todata, *fromdata;

    nx = images.nx;
    ny = images.ny;
    nimages = images.nz;

    if (xcent <= 0) xcent = (nx - 1) / 2;
    if (ycent <= 0) ycent = (ny - 1) / 2;

    /* image data is stored in row-major */
    height = croppedimages.ny;
    length = croppedimages.nx;
    ldx = length;
    ldy = height;

    ri = (length - 1) / 2;
    hi = (height - 1) / 2;

    fromdata = images.data;
    todata = croppedimages.data;
    if (!fromdata || !todata) {
        fprintf(stderr, "arecCropImages: input images not properly allocated!\n");
        exit(1);
    }

    for (k = 0; k < nimages; k++) {
        for (j = 0; j < height; j++) {
            for (i = 0; i < length; i++) {
                if (ycent - hi + j < 0 || ycent - hi + j >= ny || xcent - ri + i < 0 ||
                    xcent - ri + i >= nx) {
                    printf("CropStack: error\n");
                    printf("k = %d, j = %d, i = %d\n", k, j, i);
                    printf("x = %d, y = %d, z = %d\n", xcent - ri + i, ycent - hi + j, k);
                    printf("length = %d, height = %d\n", length, height);
                    printf("Cropping (%d,%d)->(%d,%d) \n", nx, ny, length, height);
                    printf("Center (%d,%d)\n", xcent, ycent);
                    exit(1);
                }
                todata(i, j, k) = fromdata(xcent - ri + i, ycent - hi + j, k);
            } /* end for i */
        }     /* end for j */
    }         /* end for k */
}

#undef fromdata
#undef todata

int ImageMergeYDistZ(MPI_Comm comm, arecImage imagein, arecImage imageout) {
    float *dataloc, *fuldata;
    int locsize, iangloc;
    int status = 0;

    int ncpus, mypid, i, nxloc, nyloc, nyglb, nangs, nrem, iang;
    int *psize, *nbase, *targetids, *nanglocs;

    MPI_Comm_size(comm, &ncpus);
    MPI_Comm_rank(comm, &mypid);

    psize = (int *)calloc(ncpus, sizeof(int));
    nbase = (int *)calloc(ncpus, sizeof(int));

    nxloc = imagein.nx;
    nyloc = imagein.ny;
    nangs = imagein.nz;
    nyglb = imageout.ny;

    nrem = nangs % ncpus;
    nanglocs = (int *)calloc(ncpus, sizeof(int));
    for (i = 0; i < ncpus; i++) {
        nanglocs[i] = nangs / ncpus;
        if (i < nrem) nanglocs[i]++;
    }

    targetids = (int *)calloc(ncpus, sizeof(int));
    for (i = 0; i < ncpus; i++)
        targetids[i] = i;

    locsize = nxloc * nyloc;

    /* image is viewed as column majored */
    for (i = 0; i < ncpus; i++) {
        psize[i] = locsize;
        if (i < nyglb % ncpus) psize[i] = psize[i] + nxloc;
    }
    nbase[0] = 0;
    for (i = 1; i < ncpus; i++)
        nbase[i] = nbase[i - 1] + psize[i - 1];

    dataloc = imagein.data;
    fuldata = imageout.data;
    iang = 0;
    for (i = 0; i < ncpus; i++) {
        for (iangloc = 0; iangloc < nanglocs[i]; iangloc++) {

            status = MPI_Gatherv(&dataloc[nxloc * nyloc * iang], locsize, MPI_FLOAT,
                                 &fuldata[nxloc * nyglb * iangloc], psize, nbase, MPI_FLOAT,
                                 targetids[i], comm);
            iang++;
            MPI_Barrier(comm);
        }
    }
    free(psize);
    free(nbase);
    free(targetids);
    free(nanglocs);

    return status;
}

void arecImageTakeLog(arecImage image) {
    /* take the natural log of the image intensity */
    int nx, ny, nz, nnz, i;

    if (image.is_cyl != 1) {
        nx = image.nx;
        ny = image.ny;
        nz = image.nz;
        for (i = 0; i < nx * ny * nz; i++)
            if (image.data[i] > 0) {
                image.data[i] = -log(image.data[i]);
            } else {
                image.data[i] = 0.0;
            }
    } else {
        nnz = image.nnz;
        for (i = 0; i < nnz; i++)
            if (image.data[i] > 0) {
                image.data[i] = -log(image.data[i]);
            } else {
                image.data[i] = 0.0;
            }
    }
}

/*-----------------------------------------------*/
int arecCCImages(MPI_Comm comm, arecImage images1, arecImage images2, float *sx, float *sy) {
    /* cross correlating two sets of images pairwise */
    int nx1, ny1, nz1, nx2, ny2, nz2;
    float *data1, *data2, *cccoefs;
    float *f_data1, *f_data2;
    int statusloc = 0, status = 0;
    int i;

    nx1 = images1.nx;
    ny1 = images1.ny;
    nz1 = images1.nz;
    data1 = images1.data;

    nx2 = images2.nx;
    ny2 = images2.ny;
    nz2 = images2.nz;
    data2 = images2.data;

    if (nx1 != nx2 || ny1 != ny2 || nz1 != nz2) {
        statusloc = -1;
        fprintf(stderr, "arecCCImages: mismatch in dimension:\n");
        fprintf(stderr, "nx1,nx2,ny1,ny2,nz1,nz2=%d, %d, %d, %d, %d, %d\n", nx1, nx2, ny1, ny2, nz1,
                nz2);
        MPI_Allreduce(&statusloc, &status, 1, MPI_INT, MPI_SUM, comm);
        return status;
    }

    cccoefs = (float *)fftwf_malloc(nx1 * ny1 * sizeof(float));
    f_data1 = (float *)malloc(nx1 * ny1 * sizeof(float));
    f_data2 = (float *)malloc(nx1 * ny1 * sizeof(float));
    for (i = 0; i < nz1; i++) {

        blackmanHarris_filter_normalize(&data1[nx1 * ny1 * i], f_data1, nx1, ny1);
        blackmanHarris_filter_normalize(&data2[nx2 * ny2 * i], f_data2, nx2, ny2);
        // blackmanHarris_filter(&data1[nx1*ny1*i],f_data1,nx1,ny1);
        // blackmanHarris_filter(&data2[nx2*ny2*i],f_data2,nx2,ny2);
        // ccorr2d(nx1, ny1, f_data1,f_data2, cccoefs);
        ccorr2d(nx1, ny1, &data1[nx1 * ny1 * i], &data2[nx2 * ny2 * i], cccoefs);
        /* peaksearch(nx1, ny1, cccoefs, &sx[i], &sy[i]); */
        // printf("Slic  %d with meanRaw (%4.2f,%4.2f) ",i,getArrayMean(&data1[nx1*ny1*i],
        // nx1*ny2),getArrayMean(&data2[nx1*ny1*i], nx2*ny2)); printf("Slic  %d with meanFiltered
        // (%4.2f,%4.2f) ",i,getArrayMean(f_data1, nx1*ny2),getArrayMean(f_data2, nx2*ny2));
        // printf("cccoefs[0]=%4.2f \n",cccoefs[0]);
        speak(nx1, ny1, cccoefs, &sx[i], &sy[i]);
    }
    fftwf_free(cccoefs);
    free(f_data1);
    free(f_data2);

    return status;
}

float getMinValue(float *data, int N) {
    float val = data[0];
    for (int i = 0; i < N; i++)
        if (data[i] < val) val = data[i];

    return val;
}

float getMaxValue(float *data, int N) {
    float val = data[0];
    for (int i = 0; i < N; i++)
        if (data[i] > val) val = data[i];
    return val;
}
void savePolarImage(float *data, int X, int Y, const char *fname) {
    int ix, iy;
    char *pixels = new char[X * Y * 3];
    // get vals for scaling
    double a = getMinValue(data, X * Y);
    double b = getMaxValue(data, X * Y);
    for (ix = 0; ix < X; ix++) {
        for (iy = 0; iy < Y; iy++) {
            double value = data[X * iy + ix];
            // Profile is saved in reverse order (horizontal Z)
            pixels[3 * (X * iy + ix) + 0] = (int)(255 * (value - a) / (b - a));
            pixels[3 * (X * iy + ix) + 1] = (int)(255 * (value - a) / (b - a));
            pixels[3 * (X * iy + ix) + 2] = (int)(255 * (value - a) / (b - a));
        }
    }

    // write_raw_pnm(fname, pixels, X, Y);
    int w = X;
    int h = Y;
    FILE *f;
#pragma warning(suppress : 4996)
    f = fopen(fname, "wb");
    if (!f)
        printf("Ouch!  Cannot create file.\n");
    else {
        int row;

        fprintf(f, "P6\n");
        fprintf(f, "# CREATOR: AREC PREALIGN\n");
        fprintf(f, "%d %d\n", w, h);
        fprintf(f, "255\n");

        for (row = 0; row < h; row++)
            fwrite(pixels + (row * w * 3), 1, 3 * w, f);

        fclose(f);
    }
}

int findMaxIndex(float *data, int N) {
    int i;
    float max_val = data[0];
    int max_ind = 0;
    for (i = 1; i < N; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_ind = i;
        }
    }
    return max_ind;
}

float findPeak1d(int length, float *data) {
    double plist[7];
    double pos;
    int k;
    int maxind = findMaxIndex(data, length);
    /* fit a polynomial around the peak, and return the position of
    the maximum of the polynomial */
    for (k = -3; k <= 3; k++) {
        plist[k + 3] = data[(maxind + k + length) % length];
    }
    PRB1D(plist, 7, &pos); /* fit with a polynomial and find its maximizer */
    return (pos + maxind);
}

float getCoord_blerp_raw(arecImage image, float x, float y, int z) {
    int nx = image.nx;
    int ny = image.ny;
    int xbase = (int)x;
    int ybase = (int)y;
    double xFraction = x - xbase;
    double yFraction = y - ybase;
    if (xFraction < 0.0) xFraction = 0.0;
    if (yFraction < 0.0) yFraction = 0.0;
    double lowerLeft = image.data[nx * (z * ny + ybase) + xbase];
    double lowerRight = image.data[nx * (z * ny + ybase) + xbase + 1];
    double upperRight = image.data[nx * (z * ny + ybase + 1) + xbase + 1];
    double upperLeft = image.data[nx * (z * ny + ybase + 1) + xbase];
    double upperAverage = upperLeft + xFraction * (upperRight - upperLeft);
    double lowerAverage = lowerLeft + xFraction * (lowerRight - lowerLeft);
    return lowerAverage + yFraction * (upperAverage - lowerAverage);
}

void getRadialProfile(arecImage image, int z, float *out, int length, int R) {
    int nx, ny;
    double x_c, y_c;
    int i, ri;

    nx = image.nx;
    ny = image.ny;
    x_c = 0.5 * (nx - 1);
    y_c = 0.5 * (ny - 1);

    for (i = 0; i < length; i++)
        out[i] = 0.0;
    for (i = 0; i < length; i++) {
        double alpha = 2 * piFunc() * i / length;
        for (ri = 0; ri < R; ri++) {
            out[i] +=
                getCoord_blerp_raw(image, x_c + cos(alpha) * ri, y_c + sin(alpha) * ri, z) * ri;
        }
    }
}

int arecRotCCImages(MPI_Comm comm, arecImage images1, arecImage images2, int r2, float *angles) {
    int nx, ny, nz, n_alpha;
    int i;
    float *I_alpha1, *I_alpha2, *cccoefs;
    int statusloc = 0, status = 0;

    nx = images1.nx;
    ny = images1.ny;
    nz = images1.nz;

    if (nx != images2.nx || ny != images2.ny || nz != images2.nz) {
        statusloc = -1;
        fprintf(stderr, "arecRotCCImages2: mismatch in dimension:\n");
        fprintf(stderr, "(%d,%d,%d) (%d,%d,%d)\n", nx, ny, nz, images2.nx, images2.ny, images2.nz);
        MPI_Allreduce(&statusloc, &status, 1, MPI_INT, MPI_SUM, comm);
        return status;
    }

    n_alpha = (int)(r2 * 2 * piFunc());
    I_alpha1 = (float *)fftwf_malloc(n_alpha * sizeof(float));
    I_alpha2 = (float *)fftwf_malloc(n_alpha * sizeof(float));
    cccoefs = (float *)fftwf_malloc(n_alpha * sizeof(float));
    for (i = 0; i < nz; i++) {

        getRadialProfile(images1, i, I_alpha1, n_alpha, r2);
        getRadialProfile(images2, i, I_alpha2, n_alpha, r2);

        ccorr1d(n_alpha, I_alpha1, I_alpha2, cccoefs);
        angles[i] = findPeak1d(n_alpha, cccoefs);
        // index to rad and fix phase shift
        angles[i] *= 2 * piFunc() / n_alpha; // in rad
        if (angles[i] > piFunc()) angles[i] -= 2.0 * piFunc();
        angles[i] *= 180.0 / piFunc(); // to deg
        if (fabs(angles[i]) > 45) {
            printf("Angle %4.2f too large, probably bad CC, do nothing \n", angles[i]);
            angles[i] = 0.0;
        }
    }

    fftwf_free(I_alpha1);
    fftwf_free(I_alpha2);
    fftwf_free(cccoefs);

    return status;
}

float lerp(float s, float e, float t) { return s + (e - s) * t; }
float blerp(float c00, float c10, float c01, float c11, float tx, float ty) {
    return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}
float bilinear_circ(float X, float Y, float *data, int data_X, int data_Y) {
    X += data_X;
    Y += data_Y;
    int gxi = ((int)X) % data_X;
    int gyi = ((int)Y) % data_Y;
    int gxip = (gxi + 1) % data_X;
    int gyip = (gyi + 1) % data_Y;
    float dx = X - (int)X;
    float dy = Y - (int)Y;

    // printf("(%4.2f,%4.2f) needs support ((%d,%d),(%d,%d))\n",X,Y,gxi,gyi,gxip,gyip);

    float c00 = data[gyi * data_X + gxi];
    float c10 = data[gyi * data_X + gxip];
    float c01 = data[gyip * data_X + gxi];
    float c11 = data[gyip * data_X + gxip];

    return blerp(c00, c10, c01, c11, dx, dy);
}

void bilinear_shift(int nx, int ny, float *imagein, float *imageout, float sx, float sy) {
    int i, j;
    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
            imageout[j * nx + i] = bilinear_circ(-sx + i, -sy + j, imagein, nx, ny);
        }
    }
}
void arecShiftImages_blerp(arecImage images, float *sx, float *sy) {
    /* apply circular shifts to images */
    int nx, ny, nslices;
    int i, j;
    float *data, *temp;

    nx = images.nx;
    ny = images.ny;
    nslices = images.nz;
    data = images.data;
    temp = (float *)calloc(nx * ny, sizeof(float));
    for (j = 0; j < nslices; j++) {
        bilinear_shift(nx, ny, &data[nx * ny * j], temp, -sx[j], -sy[j]);
        for (i = 0; i < nx * ny; i++)
            data[nx * ny * j + i] = temp[i];
    }
    free(temp);
}

/*-----------------------------------------------*/
void arecShiftImages(arecImage images, float *sx, float *sy) {
    /* apply circular shifts to images */
    int nx, ny, nslices;
    int i, j;
    float *data, *temp;

    nx = images.nx;
    ny = images.ny;
    nslices = images.nz;
    data = images.data;
    temp = (float *)calloc(nx * ny, sizeof(float));
    for (j = 0; j < nslices; j++) {
        /* icshift2d(nx,ny,&data[nx*ny*j],temp,-sx[j],-sy[j]); */
        fshift2d(nx, ny, &data[nx * ny * j], temp, -sx[j], -sy[j]);
        for (i = 0; i < nx * ny; i++)
            data[nx * ny * j + i] = temp[i];
    }
    free(temp);
}

void arecRotateImages(arecImage *images, float *angles) {
    /* Rotate the i-th image by angles[i] using bilinear interpolation  */
    int nx, ny, nslices;
    int i, j;
    float *data, *temp;

    nx = images->nx;
    ny = images->ny;
    nslices = images->nz;

    data = images->data;
    temp = (float *)calloc(nx * ny, sizeof(float));
    for (j = 0; j < nslices; j++) {
        rotate2d(nx, ny, &data[nx * ny * j], temp, angles[j]);
        for (i = 0; i < nx * ny; i++)
            data[nx * ny * j + i] = temp[i];
    }
    free(temp);
}

void arecRotateImages_skew(arecImage *images, float *angles) {
    /*
    Rotate the images with skewing using circular shifts
    Large rotations are done with bilinear interpolation instead.
    */
    int nx, ny, nslices;
    int i, j;
    float *data;

    nx = images->nx;
    ny = images->ny;
    nslices = images->nz;

    data = images->data;

    for (j = 0; j < nslices; j++) {
        // Fore some reason the angle may be phase shifted when
        // Fix periodic shifts:
        if (angles[j] < -180) {
            printf("Angle changed from %4.2f ", angles[j]);
            angles[j] += 360;
            printf("to %4.2f \n", angles[j]);
        }
        if (angles[j] > 180) {
            printf("Angle changed from %4.2f ", angles[j]);
            angles[j] -= 360;
            printf("to %4.2f \n", angles[j]);
        }
        if (fabs(angles[j]) > 45) {
            printf("Angle %4.2f too large, probably bad CC, do nothing \n", angles[j]);
            angles[j] = 0.0;
        } else if (fabs(angles[j]) > 10) {

            printf("Angle %4.2f too large, using bilinear instead\n", angles[j]);
            float *temp = (float *)calloc(nx * ny, sizeof(float));
            rotate2d(nx, ny, &data[nx * ny * j], temp, angles[j]);
            for (i = 0; i < nx * ny; i++)
                data[nx * ny * j + i] = temp[i];
            free(temp);
        } else {
            float angle_rad = angles[j] * piFunc() / 180.0;
            shearX_circ(&data[nx * ny * j], -tan(-0.5 * angle_rad), nx, ny);
            shearY_circ(&data[nx * ny * j], sin(-angle_rad), nx, ny);
            shearX_circ(&data[nx * ny * j], -tan(-0.5 * angle_rad), nx, ny);
        }
    }
}
// arecRotateImages_skew_safe
void arecRotateImages_skew_safe(arecImage *images, arecImage *images_ref, float *angles) {
    /*
    Rotate the images with skewing using circular shifts
    Large rotations are done with bilinear interpolation instead.
    */
    int nx, ny, nslices;
    int i, j;
    float *data, *data_ref;
    float *data_backup;
    double cc1, cc2;
    double cc_sum = 0.0;
    int n_rot = 0;

    nx = images->nx;
    ny = images->ny;
    nslices = images->nz;

    data = images->data;
    data_ref = images_ref->data;
    data_backup = (float *)malloc(nx * ny * sizeof(float));

    for (j = 0; j < nslices; j++) {
        // copy data
        for (i = 0; i < nx * ny; i++)
            data_backup[i] = data[nx * ny * j + i];
        // rotate data
        float angle_rad = angles[j] * piFunc() / 180.0;
        shearX_circ(&data[nx * ny * j], -tan(-0.5 * angle_rad), nx, ny);
        shearY_circ(&data[nx * ny * j], sin(-angle_rad), nx, ny);
        shearX_circ(&data[nx * ny * j], -tan(-0.5 * angle_rad), nx, ny);

        cc1 = getNormalizedCrossCorrelationWithFilter(&data[nx * ny * j], &data_ref[nx * ny * j],
                                                      nx, ny);
        cc2 = getNormalizedCrossCorrelationWithFilter(data_backup, &data_ref[nx * ny * j], nx, ny);

        // printf("Slice %d cc %4.4f -> %4.4f  \n "  ,j,cc2,cc1);
        if (cc2 + 0.00001 < cc1) // update angles
        {
            cc_sum += cc1;
            n_rot++;
            angles[j] *= 0.5; // test if we converge towards angle
      
        } else { // backup better, restore
            for (i = 0; i < nx * ny; i++)
                data[nx * ny * j + i] = data_backup[i];
                angles[j] = 0.0;
                cc_sum += cc2;
        }
    }
    free(data_backup);

    printf("arecRotateImages_skew_safe: Mean CC=%4.4f n_rot=%d/%d\n", cc_sum / nslices, n_rot,
           nslices);
}

void reset(arecImage *image) {
    if (image->is_cyl == 0) {
        int length = image->nx * image->ny * image->nz;
        for (int i = 0; i < length; i++) {
            image->data[i] = 0.0f;
        }
    }
    if (image->is_cyl == 1) {
        int length = image->nnz;
        for (int i = 0; i < length; i++) {
            image->data[i] = 0.0f;
        }
    }
    return;
}

void soft_noneg(arecImage *image) {
    if (image->is_cyl == 0) {
        int length = image->nx * image->ny * image->nz;
        for (int i = 0; i < length; i++) {
            if (image->data[i] < 0.0f) image->data[i] *= 0.5f;
        }
    }
    if (image->is_cyl == 1) {
        int length = image->nnz;
        for (int i = 0; i < length; i++) {
            if (image->data[i] < 0.0f) image->data[i] *= 0.5f;
        }
    }
    return;
}

void print(arecImage *image) {
    if (image->is_cyl == 0) {
        printf("\n Content of cubevol \n");
        int length = image->nx * image->ny * image->nz;
        for (int i = 0; i < length; i++) {
            printf("data %d: %f \n", i, image->data[i]);
        }
    }
    if (image->is_cyl == 1) {
        printf("\n Content of cylvol \n");
        int length = image->nnz;
        for (int i = 0; i < length; i++) {
            printf("data %d: %f  \n", i, image->data[i]);
        }
    }
    return;
}
