#pragma once

#include "fftw3.h"
#include <omp.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "./NCXT-image/image2d.h"
#include "mpfit.h"
// utilities
using namespace std;

/* This is the private data structure which contains the data points
and their uncertainties */
struct vars_struct {
    double *x;
    double *y;
    double *ey;
};
struct tube_data {
    // fitting parameters
    double p1_left, p2_left, p1_err_left;
    double p1_right, p2_right, p1_err_right;
    // limits
    int min_left, max_right;
    double COM;

    tube_data() {}
};

// 1D ARRAY FUNCTIONS
template <class T> T getMeanValue(T *data, size_t N) {
    T val = 0.0;
    for (size_t i = 0; i < N; i++) {
        val += data[i];
    }
    return val / N;
}
template <class T> T getMinValue(T *data, size_t N) {
    T val = data[0];
    for (size_t i = 0; i < N; i++)
        if (data[i] < val) val = data[i];

    return val;
}
template <class T> T getMaxValue(T *data, size_t N) {
    T val = data[0];
    for (size_t i = 0; i < N; i++)
        if (data[i] > val) val = data[i];
    return val;
}
template <class T> T getMaxAbs(T *data, size_t N) {
    T minval = getMinValue(data, N);
    T maxval = getMaxValue(data, N);
    return max(abs(minval), abs(maxval)); // overloaded in c++ NOT in c
}
template <class T> void writeArray(T *data, char *file, size_t N) {
    ofstream myfile;
    stringstream filename;
    filename << file;
    myfile.open(filename.str().c_str());
    for (int i = 0; i < N; i++) {
        myfile << data[i] << "\n";
    }
    myfile.close();
}
int findMaxIndex(const float *data, size_t N);

// SLOPE FIT
void printresult(double const *x, mp_result *result);
int rotationfunc(int m, int n, double *p, double *dy, double **dvec, void *vars);
int fitRotation(std::vector<double> &x, std::vector<double> &y, std::vector<double> &ey,
                std::vector<double> &p, double &norm, bool verb);
void linFitEdgeWerr(const int *data, int N, double &p1, double &p2, double &p1_err);
void ransac(const int *Y, int size_Y, int iter, double TH, double &p1, double &p2);
void cleanEdge(int *Y, int size_Y, double TH, double p1, double p2);

// IMAGE OPERATORS
typedef std::vector<image2d<float>> image_stack;

// AREC STUFF
void PRB1D(double const *B, int NPOINT, double *POS);
void ccorr1d(int length, float *x, float *y, float *c);

// READ / WRITE
std::vector<double> loadAngles(std::string filename);
void write_raw_pnm(const char *fname, char const *pixels, int w, int h);
void saveProfile(image2d<float> img, char const *fname);
void writeInput_quick(const char *dataname, const char *anglename);
void writeInput(const char *dataname, const char *anglename);
void writeAngles(const char *anglename, vector<double> angles);
void writePrealignment(const char *file, double rot, vector<int> sx, vector<int> sx2,
                       vector<int> sy);
void writeArray(double *data, char *file, int N);
std::vector<double> loadAngles(std::string filename);
void writeAngles(const char *anglename, vector<double> angles);

// PROFILE ALIGN
float getLineMean(image_stack &img, int z, int y, int x_min, int x_max);
void circShiftX(float *data, int length, int shift);
double findMaxCC_fit(float const *p_cccoefs, int length);

// Convolutions
vector<float> getGaussianDerivateKernel(double n_sigma, double sigma);
vector<float> convolve_valid(float *data, const float *kernel, float fill_val, int len,
                             int len_halfkernel);
void convolve_valid_inplace(float *data, float *kernel, float fill_val, int len,
                            int len_halfkernel);
