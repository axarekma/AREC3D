#pragma once

#include <mpi.h>

typedef struct Image {
    /* cubic 3D image */
    int nx;
    int ny;
    int nz;
    /* cylindrincal 3D image */
    double radius;
    int height;
    int nnz;
    int nrays;
    int *cord;
    int is_cyl; /* 1 for true, 0 for false */
    float *data;
} arecImage;

#define cord(i, j) cord[(j)*2 + (i)]
#define cube(i, j, k) cube[(((k)) * ny + (j)) * nx + (i)]
#define cylval(i) cylval[(i)]

// #define min(x1, x2) ((x1) > (x2) ? (x2) : (x1))
// #define max(x1, x2) ((x1) > (x2) ? (x1) : (x2))

int arecAllocateCBImage(arecImage *cbimage, int nx, int ny, int nz);
int arecAllocateCylImage(arecImage *cylimage, double radius, int height);
int ImageCB2Cyl(arecImage cbimage, arecImage cylimage, int xcent, int ycent, double radius,
                int height);
int ImageCyl2CB(arecImage cylimage, arecImage cbimage, int nx, int ny, int nz);
void arecImageFree(arecImage *image);
void arecCropImages(arecImage images, arecImage croppedimages, int xcent, int ycent);
void arecImageTakeLog(arecImage image);
int ImageMergeYDistZ(MPI_Comm comm, arecImage imagein, arecImage imageout);
int arecCCImages(MPI_Comm comm, arecImage images1, arecImage images2, float *sx, float *sy);
int arecRotCCImages(MPI_Comm comm, arecImage images1, arecImage images2, int r2, float *angle);

float lerp(float s, float e, float t);
float blerp(float c00, float c10, float c01, float c11, float tx, float ty);
float bilinear_circ(float X, float Y, float *data, int data_X, int data_Y);
void bilinear_shift(int nx, int ny, float *imagein, float *imageout, float sx, float sy);
void arecShiftImages_blerp(arecImage images, float *sx, float *sy);

void arecShiftImages(arecImage images, float *sx, float *sy);
void arecRotateImages(arecImage *images, float *angles);
void arecRotateImages_skew(arecImage *images, float *angles);
void arecRotateImages_skew_safe(arecImage *images, arecImage *images_ref, float *angles);

float getCoord_blerp_raw(arecImage image, float x, float y, int z);
void getRadialProfile(arecImage image, int z, float *out, int length, int R);

float findPeak1d(int length, float *data);
int findMaxIndex(float *data, int N);

void soft_noneg(arecImage *image);
void reset(arecImage *image);
void print(arecImage *image);
