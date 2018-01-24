#pragma once

void periodicSmoothing(arecImage img);
void makePeriodicSlice(int nx, int ny, float *data, float *out);
void periodicDecomposition(arecImage img);

void circShiftX(float *data, int length, int shift);
void circShiftY(float *data, int nx, int length, int shift, int x);
void circShift2D(float *data, int nx, int ny, int shift, int index, int dim);
void shearX_circ(float *img, float shear, int nx, int ny);
void shearY_circ(float *img, float shear, int nx, int ny);

void blackmanHarris_filter(float *in, float *out, int nx, int ny);
void blackmanHarris_filter_normalize(float *in, float *out, int nx, int ny);
void tukey_filter(float *in, float *out, int nx, int ny, double alpha);
void tukey_filter_inplace(float *in, int nx, int ny, double alpha);

double getNormalizedCrossCorrelation(float *im1, float *im2, int nx, int ny);
double getNormalizedCrossCorrelationWithFilter(float *im1, float *im2, int nx, int ny);

float getArrayMean(float *data, int N);
