#pragma once

#include <mpi.h>

#include "arecConstants.h"
#include "arecImage.h"

void ccorr1d(int nx, float *x, float *y, float *c);
void ccorr2d(int nx, int ny, float *x, float *y, float *c);
void ccorr2dfix(int nx, int ny, float *x, float *y, float *c);
void peaksearch(int nx, int ny, float *cccoefs, int *ix, int *iy);
void icshift2d(int nx, int ny, float *imagein, float *imageout, int ix, int iy);
void fshift2d(int nx, int ny, float *imagein, float *imageout, float sx, float sy);
void iaccshifts(MPI_Comm comm, int nzloc, int *shiftin, int *shiftout);
void accshifts(MPI_Comm comm, int nzloc, float *shiftin, float *shiftout);
void align2dstack(MPI_Comm comm, arecImage imagestack, arecImage alignedstack, int lcut, int rcut,
                  int fudgefactor, int iflip);
void getscf(int nx, int ny, float *x, float *y);
void cart2po(int nx, int ny, float *cdata, int nang, int r1, int r2, float *pdata);
void rotate2d(int NROW, int NSAM, float *imagein, float *imageout, float THETA);
double QUADRI(double XX, double YY, int NXDATA, int NYDATA, float *FDATA);
void to_polar(int NSAM, int NROW, int R1, int R2, float *X, float *Y);
void PRB1D(double *B, int NPOINT, double *POS);
void quad2dfit(float *Z, float *XSH, float *YSH, float *PEAKV);
void speak(int nx, int ny, float *data, float *xt, float *yt);
