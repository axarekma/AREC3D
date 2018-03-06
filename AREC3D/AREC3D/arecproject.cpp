#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// #include "arecConstants.h"
#include "arecImage.h"
#include "arecproject.h"

inline int double2int(double d) {
    d += 6755399441055744.0;
    return reinterpret_cast<int &>(d);
}

//#define cord(i, j) cord[((j)-1) * 2 + (i)-1]
#define x(i) x[(i)-1]
#define y(i, j, k) y[nx * ny * (k) + ((j)-1) * nx + (i)-1]

int arecProject2D(arecImage cylvol, float *angles, int nangles, arecImage *projstack) {
    /*
    purpose:  y <--- proj(x)
    input  :  volsize  the size (nx,ny,nz) of the volume
    nrays    number of rays within the compact cylinderincal
    representation
    cord     the coordinates of the first point in each ray
    x        3d input volume
    y        2d output image
    */

    int nx, ny, nrays, nnz, nnz0, xcent, zcent;
    int ia, iqx, iqy, i, xc, zc, radius;
    float ct, dipx, dipx1m, xb;
    int status = 0;
    float *x, *y;
    int *cord;
    double cosphi, sinphi;

    // float sx, sy;

    if (!cylvol.is_cyl) {
        fprintf(stderr, "invalid input volume format. Must be cylindrical!\n");
        status = 1;
        return status;
    }

    /* no translation at the moment */
    // sx = 0.0;
    // sy = 0.0;

    nx = cylvol.nx;
    ny = cylvol.ny;
    x = cylvol.data;

    radius = cylvol.radius;
    nrays = cylvol.nrays;
    nnz0 = cylvol.nnz;
    cord = cylvol.cord;

    xcent = radius + 1;
    zcent = radius + 1;

    y = projstack->data;
    if (projstack->nx != nx || projstack->ny != ny) {
        fprintf(stderr, "arecProject2D: inconsistent dimension!\n");
        status = 2;
        return status;
    }
    for (i = 0; i < nx * ny * projstack->nz; i++)
        y[i] = 0.0;

    for (ia = 0; ia < nangles; ia++) {
        cosphi = cos(angles[ia]);
        sinphi = sin(angles[ia]);
        nnz = 0;
        for (i = 1; i <= nrays; i++) {
            xc = cord(1, i) - xcent;
            zc = cord(2, i) - zcent;
            xb = xc * cosphi + zc * sinphi + xcent;
            // only need 1D interpolation, compute weights
            iqx = floor(xb);
            dipx = xb - iqx;
            dipx1m = 1.0 - dipx;
            for (iqy = 1; iqy <= ny; iqy++) {
                ct = x(nnz + iqy);
                // optimize later
                if (iqx > 0 && iqx <= nx) y(iqx, iqy, ia) += dipx1m * ct;
                if (iqx + 1 > 0 && iqx + 1 <= nx) y(iqx + 1, iqy, ia) += dipx * ct;
            } /* end for iqx */
            nnz = nnz + ny;
        } /* end for i */
        if (nnz0 != nnz) {
            fprintf(stderr, "arecProject2D: problem with nnz count!\n");
            status = 2;
            break;
        }
    } /* end for ia */

    return status;
}

#undef x
#undef y

    /*-----------------------------------------------------------------*/

#define y(i) y[(i)-1]
#define x(i, j, k) x[nx * ny * (k) + ((j)-1) * nx + (i)-1]

int arecBackProject2D(arecImage images, float *angles, int nangles, arecImage *bvol) {
    int nx, ny, nz, i, j, ia, iqx, zc, xc, nnz, nrays, radius, xcent, zcent, nnz0;
    float xb, dx, dx1m, sx, sy;
    int status = 0;
    int *cord;
    float *x, *y;
    double cosphi, sinphi;

    nx = images.nx;
    ny = images.ny;
    nz = images.nz;
    if (nz != nangles) {
        status = 1;
        return status;
    }
    x = images.data;

    if (bvol->nx != nx || bvol->ny != ny) {
        status = -1;
        fprintf(stderr, "arecBackProject2D: image and volume sizes don't match\n");
        return status;
    }

    radius = bvol->radius;
    nrays = bvol->nrays;
    nnz0 = bvol->nnz;
    cord = bvol->cord;
    if (!bvol->is_cyl) {
        status = -1;
        fprintf(stderr, "arecBackProject2D: cylindrical volume ");
        fprintf(stderr, "not properly set!\n");
        return status;
    }
    y = bvol->data;
    for (i = 0; i < nnz0; i++)
        y[i] = 0.0;

    xcent = radius + 1;
    zcent = radius + 1;

    sx = 0.0;
    sy = 0.0;

    for (ia = 0; ia < nangles; ia++) {
        cosphi = cos(angles[ia]);
        sinphi = sin(angles[ia]);
        nnz = 0;
        for (i = 1; i <= nrays; i++) {
            xc = cord(1, i) - xcent;
            zc = cord(2, i) - zcent;
            xb = xc * cosphi + zc * sinphi + xcent + sx;

            iqx = floor(xb);
            dx = xb - iqx;
            dx1m = 1.0 - dx;

            if (iqx <= nx && iqx >= 1)
                for (j = 1; j <= ny; j++)
                    y(nnz + j) += dx1m * x(iqx, j, ia);

            if (iqx + 1 <= nx && iqx + 1 >= 1)
                for (j = 1; j <= ny; j++)
                    y(nnz + j) += dx * x(iqx + 1, j, ia);

            nnz = nnz + ny;
        } /* end for i */
        if (nnz0 != nnz) {
            fprintf(stderr, "arecProject2D: problem with nnz count!\n");
            status = 2;
            break;
        }
    } /* end for ia */

    return status;
}

#undef x
#undef y

/*------------- KAISER BESSEL BLOB IMPLEMENTATION --------------*/
// Hardcoded LUT for Kaiser-Bessel Blob Interpolatiuon
// calculated by convolution with a 1-pixel size box.
// A bit ugly, sorry :)
#define LUT_SIZE 1000
#define LUT_RANGE 3.0
float DR = LUT_RANGE / (LUT_SIZE - 1);
float LUT[LUT_SIZE] = {
    0.64949,     0.64949,     0.64946,     0.64943,     0.64937,     0.64931,     0.64922,
    0.64914,     0.64902,     0.6489,      0.64876,     0.64861,     0.64844,     0.64826,
    0.64806,     0.64785,     0.64762,     0.64738,     0.64712,     0.64686,     0.64657,
    0.64628,     0.64595,     0.64563,     0.64529,     0.64494,     0.64456,     0.64418,
    0.64377,     0.64337,     0.64293,     0.6425,      0.64203,     0.64157,     0.64108,
    0.64059,     0.64007,     0.63955,     0.639,       0.63845,     0.63788,     0.6373,
    0.6367,      0.63609,     0.63546,     0.63483,     0.63417,     0.63351,     0.63283,
    0.63214,     0.63143,     0.63072,     0.62998,     0.62924,     0.62847,     0.6277,
    0.62691,     0.62612,     0.6253,      0.62448,     0.62363,     0.62278,     0.62191,
    0.62104,     0.62014,     0.61925,     0.61832,     0.6174,      0.61645,     0.6155,
    0.61453,     0.61355,     0.61256,     0.61156,     0.61054,     0.60951,     0.60846,
    0.60742,     0.60635,     0.60527,     0.60418,     0.60308,     0.60197,     0.60085,
    0.5997,      0.59856,     0.5974,      0.59623,     0.59504,     0.59386,     0.59265,
    0.59143,     0.5902,      0.58897,     0.58772,     0.58646,     0.58518,     0.58391,
    0.58261,     0.58131,     0.58,        0.57868,     0.57734,     0.576,       0.57464,
    0.57328,     0.5719,      0.57052,     0.56912,     0.56772,     0.5663,      0.56489,
    0.56345,     0.56201,     0.56055,     0.5591,      0.55762,     0.55615,     0.55466,
    0.55316,     0.55165,     0.55014,     0.54861,     0.54709,     0.54554,     0.544,
    0.54243,     0.54087,     0.5393,      0.53772,     0.53612,     0.53453,     0.53292,
    0.53131,     0.52969,     0.52806,     0.52642,     0.52478,     0.52313,     0.52148,
    0.51981,     0.51814,     0.51646,     0.51478,     0.51308,     0.51139,     0.50968,
    0.50797,     0.50625,     0.50453,     0.50279,     0.50106,     0.49932,     0.49757,
    0.49581,     0.49406,     0.49229,     0.49052,     0.48874,     0.48696,     0.48518,
    0.48339,     0.48159,     0.47979,     0.47798,     0.47617,     0.47435,     0.47254,
    0.47071,     0.46888,     0.46705,     0.46521,     0.46337,     0.46153,     0.45968,
    0.45783,     0.45597,     0.45411,     0.45224,     0.45038,     0.44851,     0.44664,
    0.44476,     0.44288,     0.441,       0.43911,     0.43722,     0.43533,     0.43344,
    0.43155,     0.42965,     0.42775,     0.42585,     0.42394,     0.42203,     0.42013,
    0.41822,     0.41631,     0.41439,     0.41248,     0.41056,     0.40864,     0.40673,
    0.40481,     0.40288,     0.40096,     0.39904,     0.39712,     0.39519,     0.39327,
    0.39134,     0.38942,     0.38749,     0.38556,     0.38364,     0.38171,     0.37978,
    0.37786,     0.37593,     0.374,       0.37208,     0.37015,     0.36823,     0.3663,
    0.36438,     0.36246,     0.36053,     0.35861,     0.35669,     0.35477,     0.35286,
    0.35094,     0.34902,     0.34711,     0.3452,      0.34329,     0.34138,     0.33947,
    0.33757,     0.33566,     0.33376,     0.33186,     0.32996,     0.32807,     0.32617,
    0.32428,     0.3224,      0.32051,     0.31863,     0.31674,     0.31487,     0.31299,
    0.31112,     0.30925,     0.30739,     0.30552,     0.30366,     0.3018,      0.29995,
    0.2981,      0.29625,     0.29441,     0.29257,     0.29073,     0.2889,      0.28707,
    0.28525,     0.28343,     0.28161,     0.27979,     0.27799,     0.27618,     0.27438,
    0.27258,     0.27079,     0.269,       0.26722,     0.26544,     0.26367,     0.26189,
    0.26013,     0.25837,     0.25661,     0.25486,     0.25312,     0.25137,     0.24964,
    0.2479,      0.24618,     0.24446,     0.24274,     0.24103,     0.23933,     0.23762,
    0.23593,     0.23424,     0.23256,     0.23087,     0.2292,      0.22753,     0.22587,
    0.22421,     0.22257,     0.22092,     0.21928,     0.21765,     0.21602,     0.2144,
    0.21278,     0.21117,     0.20957,     0.20797,     0.20638,     0.20479,     0.20322,
    0.20164,     0.20008,     0.19851,     0.19696,     0.19541,     0.19387,     0.19233,
    0.19081,     0.18928,     0.18777,     0.18626,     0.18476,     0.18326,     0.18177,
    0.18028,     0.17881,     0.17734,     0.17588,     0.17442,     0.17297,     0.17152,
    0.17009,     0.16866,     0.16724,     0.16582,     0.16442,     0.16301,     0.16162,
    0.16023,     0.15885,     0.15747,     0.15611,     0.15474,     0.15339,     0.15204,
    0.15071,     0.14937,     0.14805,     0.14673,     0.14542,     0.14411,     0.14282,
    0.14152,     0.14024,     0.13896,     0.1377,      0.13643,     0.13518,     0.13393,
    0.1327,      0.13146,     0.13024,     0.12901,     0.12781,     0.1266,      0.1254,
    0.12421,     0.12303,     0.12185,     0.12069,     0.11952,     0.11837,     0.11722,
    0.11608,     0.11495,     0.11382,     0.1127,      0.11159,     0.11049,     0.10939,
    0.1083,      0.10722,     0.10614,     0.10507,     0.10401,     0.10296,     0.10191,
    0.10087,     0.099833,    0.09881,     0.097788,    0.096779,    0.095771,    0.094776,
    0.093782,    0.092801,    0.09182,     0.090854,    0.089887,    0.088935,    0.087982,
    0.087043,    0.086104,    0.085179,    0.084254,    0.083343,    0.082432,    0.081534,
    0.080637,    0.079753,    0.078869,    0.077998,    0.077128,    0.076271,    0.075414,
    0.07457,     0.073726,    0.072896,    0.072066,    0.071249,    0.070432,    0.069628,
    0.068824,    0.068033,    0.067242,    0.066464,    0.065686,    0.064921,    0.064156,
    0.063404,    0.062652,    0.061912,    0.061173,    0.060446,    0.059719,    0.059005,
    0.05829,     0.057588,    0.056886,    0.056197,    0.055507,    0.05483,     0.054152,
    0.053487,    0.052821,    0.052168,    0.051515,    0.050873,    0.050232,    0.049602,
    0.048972,    0.048354,    0.047736,    0.047129,    0.046523,    0.045928,    0.045333,
    0.044749,    0.044165,    0.043593,    0.04302,     0.042459,    0.041897,    0.041347,
    0.040796,    0.040257,    0.039717,    0.039188,    0.038659,    0.038141,    0.037623,
    0.037115,    0.036607,    0.03611,     0.035612,    0.035125,    0.034638,    0.034161,
    0.033684,    0.033217,    0.03275,     0.032292,    0.031835,    0.031388,    0.03094,
    0.030502,    0.030065,    0.029636,    0.029208,    0.028789,    0.02837,     0.02796,
    0.02755,     0.02715,     0.026749,    0.026357,    0.025966,    0.025583,    0.0252,
    0.024826,    0.024452,    0.024086,    0.023721,    0.023364,    0.023007,    0.022658,
    0.02231,     0.021969,    0.021629,    0.021296,    0.020964,    0.02064,     0.020315,
    0.019999,    0.019682,    0.019373,    0.019065,    0.018763,    0.018462,    0.018168,
    0.017875,    0.017588,    0.017302,    0.017023,    0.016743,    0.016471,    0.016199,
    0.015934,    0.015669,    0.015411,    0.015152,    0.014901,    0.014649,    0.014404,
    0.014159,    0.013921,    0.013682,    0.01345,     0.013218,    0.012992,    0.012766,
    0.012547,    0.012327,    0.012113,    0.0119,      0.011692,    0.011484,    0.011282,
    0.01108,     0.010884,    0.010687,    0.010497,    0.010306,    0.010121,    0.0099352,
    0.0097552,   0.0095752,   0.0094005,   0.0092257,   0.0090561,   0.0088864,   0.0087218,
    0.0085572,   0.0083975,   0.0082377,   0.0080828,   0.0079279,   0.0077776,   0.0076274,
    0.0074817,   0.0073361,   0.0071949,   0.0070538,   0.006917,    0.0067802,   0.0066477,
    0.0065153,   0.006387,    0.0062587,   0.0061344,   0.0060102,   0.00589,     0.0057698,
    0.0056535,   0.0055372,   0.0054247,   0.0053122,   0.0052034,   0.0050946,   0.0049894,
    0.0048842,   0.0047826,   0.0046809,   0.0045827,   0.0044845,   0.0043897,   0.0042948,
    0.0042032,   0.0041116,   0.0040233,   0.0039349,   0.0038496,   0.0037643,   0.003682,
    0.0035997,   0.0035204,   0.0034411,   0.0033646,   0.0032881,   0.0032144,   0.0031407,
    0.0030697,   0.0029987,   0.0029304,   0.002862,    0.0027962,   0.0027304,   0.0026671,
    0.0026038,   0.0025428,   0.0024819,   0.0024233,   0.0023648,   0.0023085,   0.0022522,
    0.0021981,   0.002144,    0.002092,    0.00204,     0.0019901,   0.0019402,   0.0018923,
    0.0018445,   0.0017985,   0.0017526,   0.0017085,   0.0016645,   0.0016222,   0.00158,
    0.0015395,   0.0014991,   0.0014603,   0.0014216,   0.0013845,   0.0013474,   0.0013119,
    0.0012764,   0.0012424,   0.0012084,   0.001176,    0.0011435,   0.0011125,   0.0010815,
    0.0010518,   0.0010222,   0.0009939,   0.00096561,  0.00093861,  0.00091162,  0.00088587,
    0.00086012,  0.00083557,  0.00081103,  0.00078764,  0.00076426,  0.00074199,  0.00071972,
    0.00069852,  0.00067733,  0.00065716,  0.000637,    0.00061784,  0.00059867,  0.00058046,
    0.00056224,  0.00054495,  0.00052765,  0.00051124,  0.00049483,  0.00047926,  0.0004637,
    0.00044894,  0.00043419,  0.00042021,  0.00040623,  0.000393,    0.00037977,  0.00036726,
    0.00035474,  0.00034291,  0.00033108,  0.0003199,   0.00030872,  0.00029817,  0.00028762,
    0.00027766,  0.00026771,  0.00025832,  0.00024894,  0.0002401,   0.00023127,  0.00022295,
    0.00021463,  0.00020681,  0.00019899,  0.00019164,  0.00018429,  0.00017739,  0.00017049,
    0.00016402,  0.00015755,  0.00015149,  0.00014542,  0.00013975,  0.00013407,  0.00012876,
    0.00012345,  0.00011849,  0.00011353,  0.0001089,   0.00010427,  9.9947e-005, 9.5628e-005,
    9.1606e-005, 8.7583e-005, 8.3841e-005, 8.0099e-005, 7.6621e-005, 7.3143e-005, 6.9915e-005,
    6.6687e-005, 6.3695e-005, 6.0702e-005, 5.7932e-005, 5.5161e-005, 5.26e-005,   5.0038e-005,
    4.7673e-005, 4.5308e-005, 4.3128e-005, 4.0947e-005, 3.894e-005,  3.6933e-005, 3.5088e-005,
    3.3243e-005, 3.155e-005,  2.9857e-005, 2.8306e-005, 2.6755e-005, 2.5337e-005, 2.3919e-005,
    2.2624e-005, 2.133e-005,  2.0151e-005, 1.8972e-005, 1.79e-005,   1.6828e-005, 1.5855e-005,
    1.4883e-005, 1.4003e-005, 1.3123e-005, 1.2328e-005, 1.1534e-005, 1.0818e-005, 1.0102e-005,
    9.4593e-006, 8.8164e-006, 8.2406e-006, 7.6648e-006, 7.1505e-006, 6.6362e-006, 6.1784e-006,
    5.7206e-006, 5.3145e-006, 4.9083e-006, 4.5493e-006, 4.1902e-006, 3.8741e-006, 3.5579e-006,
    3.2807e-006, 3.0035e-006, 2.7615e-006, 2.5194e-006, 2.3092e-006, 2.099e-006,  1.9173e-006,
    1.7357e-006, 1.5796e-006, 1.4235e-006, 1.2903e-006, 1.157e-006,  1.0441e-006, 9.3109e-007,
    8.3605e-007, 7.4101e-007, 6.6174e-007, 5.8246e-007, 5.1697e-007, 4.5148e-007, 3.9795e-007,
    3.4443e-007, 3.0122e-007, 2.5802e-007, 2.2363e-007, 1.8924e-007, 1.6232e-007, 1.354e-007,
    1.1474e-007, 9.4067e-008, 7.8562e-008, 6.3057e-008, 5.1751e-008, 4.0445e-008, 3.2487e-008,
    2.4529e-008, 1.9175e-008, 1.3822e-008, 1.0432e-008, 7.0418e-009, 5.0691e-009, 3.0963e-009,
    2.0854e-009, 1.0746e-009, 6.5611e-010, 2.3767e-010, 1.2572e-010, 1.3772e-011, 6.8859e-012,
    7.3808e-021, 7.3797e-021, 7.3797e-021, 7.3797e-021, 7.3797e-021, 7.3795e-021, 7.3794e-021,
    7.3792e-021, 7.3789e-021, 7.3784e-021, 7.3779e-021, 7.3771e-021, 7.3762e-021, 7.3752e-021,
    7.3742e-021, 7.3729e-021, 7.3715e-021, 7.3691e-021, 7.3668e-021, 7.3654e-021, 7.3641e-021,
    7.3607e-021, 7.3574e-021, 7.3547e-021, 7.352e-021,  7.3493e-021, 7.3466e-021, 7.3425e-021,
    7.3385e-021, 7.3331e-021, 7.3277e-021, 7.3223e-021, 7.3169e-021, 7.3115e-021, 7.3061e-021,
    7.2994e-021, 7.2927e-021, 7.2859e-021, 7.2792e-021, 7.2738e-021, 7.2684e-021, 7.2657e-021,
    7.263e-021,  7.2091e-021, 7.1552e-021, 7.0905e-021, 7.0258e-021, 6.9477e-021, 6.8695e-021,
    6.8372e-021, 6.8048e-021, 6.732e-021,  6.6593e-021, 6.6027e-021, 6.5461e-021, 6.5973e-021,
    6.6485e-021, 6.7051e-021, 6.7617e-021, 6.697e-021,  6.6323e-021, 6.5757e-021, 6.5191e-021,
    6.4841e-021, 6.449e-021,  6.4248e-021, 6.4005e-021, 6.3682e-021, 6.3358e-021, 6.2873e-021,
    6.2388e-021, 6.2037e-021, 6.1687e-021, 6.1957e-021, 6.2226e-021, 6.2145e-021, 6.2064e-021,
    6.193e-021,  6.1795e-021, 6.1795e-021, 6.1795e-021, 6.1633e-021, 6.1471e-021, 6.1471e-021,
    6.1471e-021, 6.1525e-021, 6.1579e-021, 6.1485e-021, 6.1391e-021, 6.1377e-021, 6.1364e-021,
    6.1391e-021, 6.1418e-021, 6.1401e-021, 6.1384e-021, 6.1383e-021, 6.1381e-021, 6.1395e-021,
    6.1408e-021, 6.1428e-021, 6.1449e-021, 6.1449e-021, 6.1449e-021, 6.1435e-021, 6.1422e-021,
    6.1462e-021, 6.1503e-021, 6.1664e-021, 6.1826e-021, 6.1745e-021, 6.1664e-021, 6.1745e-021,
    6.1826e-021, 6.1557e-021, 6.1287e-021, 6.1422e-021, 6.1557e-021, 6.1772e-021, 6.1988e-021,
    6.1961e-021, 6.1934e-021, 6.2042e-021, 6.215e-021,  6.2311e-021, 6.2473e-021, 6.2527e-021,
    6.2581e-021, 6.2958e-021, 6.3335e-021, 6.3875e-021, 6.4414e-021, 6.4575e-021, 6.4737e-021,
    6.4144e-021, 6.3551e-021, 6.3066e-021, 6.2581e-021, 6.1395e-021, 6.0209e-021, 5.8915e-021,
    5.7621e-021, 5.7999e-021, 5.8376e-021, 6.0748e-021, 6.312e-021,  6.506e-021,  6.7001e-021,
    6.7918e-021, 6.8834e-021, 6.8726e-021, 6.8618e-021, 6.9157e-021, 6.9696e-021, 7.1637e-021,
    7.3578e-021, 7.5518e-021, 7.7459e-021, 7.9076e-021, 8.0693e-021, 8.0586e-021, 8.0478e-021,
    8.1017e-021, 8.1556e-021, 8.1017e-021, 8.0478e-021, 7.9615e-021, 7.8753e-021};

float interp1_R_MOD(float x) {
    x = fabs(x);
    x /= DR;
    int ind = (int)(x);
    x -= ind;
    return (LUT[ind] * (1.0 - x)) + (LUT[ind + 1] * x);
}

#define x(i) x[(i)]
#define y(i, j, k) y[nx * ny * (k) + ((j)) * nx + (i)]

int arecProject2D_KB(arecImage cylvol, float *angles, int nangles, arecImage *projstack) {
    /*
    purpose:  y <--- proj(x)
    input  :  volsize  the size (nx,ny,nz) of the volume
    nrays    number of rays within the compact cylinderincal
    representation
    cord     the coordinates of the first point in each ray
    x        3d input volume
    y        2d output image
    */

    int status = 0;
    float *x, *y;
    int *cord;
    double cosphi, sinphi;

    // float sx, sy;

    if (!cylvol.is_cyl) {
        fprintf(stderr, "invalid input volume format. Must be cylindrical!\n");
        status = 1;
        return status;
    }

    int const nx = cylvol.nx;
    int const ny = cylvol.ny;
    // int const nz = cylvol.nz;
    x = cylvol.data;

    double const radius = cylvol.radius;
    int const nrays = cylvol.nrays;
    int const nnz0 = cylvol.nnz;
    cord = cylvol.cord;

    double const xcent = radius;
    double const zcent = radius;

    y = projstack->data;
    if (projstack->nx != nx || projstack->ny != ny) {
        fprintf(stderr, "arecProject2D: inconsistent dimension!\n");
        status = 2;
        return status;
    }
    for (int i = 0; i < nx * ny * projstack->nz; i++)
        y[i] = 0.0;

    for (int ia = 0; ia < nangles; ia++) {
        cosphi = cos(angles[ia]);
        sinphi = sin(angles[ia]);
        int nnz = 0;
        for (int i = 0; i < nrays; i++) {
            double const xc = cord(0, i) - xcent;
            double const zc = cord(1, i) - zcent;
            double const xb = xc * cosphi + zc * sinphi + xcent;
            // take neighbouring weights /pm 1 pixel
            // iqx = (int)roundf(xb);
            int const iqx = double2int(xb); // this is much faster, make check to see if valid
            double const w0 = interp1_R_MOD(xb - (iqx - 1));
            double const w1 = interp1_R_MOD(xb - iqx);
            double const w2 = interp1_R_MOD(xb - (iqx + 1));

            for (int iqy = 0; iqy < ny; iqy++) {
                float const ct = x(nnz + iqy);
                // optimize later
                if (iqx > 0 && iqx + 1 < nx) {
                    y(iqx - 1, iqy, ia) += w0 * ct;
                    y(iqx, iqy, ia) += w1 * ct;
                    y(iqx + 1, iqy, ia) += w2 * ct;
                } else {
                    if (is_inside(iqx - 1, nx)) { y(iqx - 1, iqy, ia) += w0 * ct; }
                    if (is_inside(iqx, nx)) { y(iqx, iqy, ia) += w1 * ct; }
                    if (is_inside(iqx + 1, nx)) { y(iqx + 1, iqy, ia) += w2 * ct; }
                }

            } /* end for iqx */

            nnz = nnz + ny;
        } /* end for i */
        if (nnz0 != nnz) {
            fprintf(stderr, "arecProject2D: problem with nnz count!\n");
            status = 2;
            break;
        }
    } /* end for ia */

    return status;
}

#undef x
#undef y

    /*-----------------------------------------------------------------*/

#define y(i) y[(i)]
#define x(i, j, k) x[nx * ny * (k) + ((j)) * nx + (i)]

int arecBackProject2D_KB(arecImage images, float *angles, int nangles, arecImage *bvol) {

    int status = 0;
    int *cord;
    float *x, *y;

    int const nx = images.nx;
    int const ny = images.ny;
    int const nz = images.nz;
    if (nz != nangles) {
        status = 1;
        return status;
    }
    x = images.data;

    if (bvol->nx != nx || bvol->ny != ny) {
        status = -1;
        fprintf(stderr, "arecBackProject2D: image and volume sizes don't match\n");
        return status;
    }

    double const radius = bvol->radius;
    int const nrays = bvol->nrays;
    int const nnz0 = bvol->nnz;
    cord = bvol->cord;
    if (!bvol->is_cyl) {
        status = -1;
        fprintf(stderr, "arecBackProject2D: cylindrical volume ");
        fprintf(stderr, "not properly set!\n");
        return status;
    }
    y = bvol->data;
    for (int i = 0; i < nnz0; i++)
        y[i] = 0.0;

    double const xcent = radius;
    double const zcent = radius;

    for (int ia = 0; ia < nangles; ia++) {
        double const cosphi = cos(angles[ia]);
        double const sinphi = sin(angles[ia]);
        int nnz = 0;
        for (int i = 0; i < nrays; i++) {
            double const xc = cord(0, i) - xcent;
            double const zc = cord(1, i) - zcent;
            double const xb = xc * cosphi + zc * sinphi + xcent;

            // take neighbouring weights /pm 1 pixel
            // iqx = (int)roundf(xb);
            int const iqx = double2int(xb); // this is much faster, make check to see if valid
            double const w0 = interp1_R_MOD(xb - (iqx - 1));
            double const w1 = interp1_R_MOD(xb - iqx);
            double const w2 = interp1_R_MOD(xb - (iqx + 1));

            if (iqx > 0 && iqx + 1 < nx) { // all possible pixles inside
                for (int j = 0; j < ny; j++) {
                    y(nnz + j) += w0 * x(iqx - 1, j, ia);
                    y(nnz + j) += w1 * x(iqx, j, ia);
                    y(nnz + j) += w2 * x(iqx + 1, j, ia);
                }
            } else { // check boundaries separately
                if (is_inside(iqx - 1, nx)) {
                    for (int j = 0; j < ny; j++) {
                        y(nnz + j) += w0 * x(iqx - 1, j, ia);
                    }
                }
                if (is_inside(iqx, nx)) {
                    for (int j = 0; j < ny; j++) {
                        y(nnz + j) += w1 * x(iqx, j, ia);
                    }
                }
                if (is_inside(iqx + 1, nx)) {
                    for (int j = 0; j < ny; j++) {
                        y(nnz + j) += w2 * x(iqx + 1, j, ia);
                    }
                }
            }

            nnz = nnz + ny;
        } /* end for i */
        if (nnz0 != nnz) {
            fprintf(stderr, "arecProject2D: problem with nnz count!\n");
            status = 2;
            break;
        }
    } /* end for ia */

    return status;
}

#undef x
#undef y

int arecProject2D_SQ(arecImage const cylvol, float *angles, int nangles, arecImage *projstack) {
    /*
    purpose:  y <--- proj(x)
    input  :  volsize  the size (nx,ny,nz) of the volume
    nrays    number of rays within the compact cylinderincal
    representation
    cord     the coordinates of the first point in each ray
    x        3d input volume
    y        2d output image
    */

    int status = 0;

    if (!cylvol.is_cyl) {
        fprintf(stderr, "invalid input volume format. Must be cylindrical!\n");
        status = 1;
        return status;
    }

    int const nx = cylvol.nx;
    int const ny = cylvol.ny;
    // int const nz = cylvol.nz;

    float *x = cylvol.data;

    double const radius = cylvol.radius;
    int const nrays = cylvol.nrays;
    int const nnz0 = cylvol.nnz;

    int *cord = cylvol.cord;

    double const xcent = radius;
    double const zcent = radius;

    float *y = projstack->data;

    if (projstack->nx != nx || projstack->ny != ny) {
        fprintf(stderr, "arecProject2D: inconsistent dimension!\n");
        status = 2;
        return status;
    }
    reset(projstack);

    // loops
    int ia, i, iqx, iqy;
    int nnz, ind_a, ind_y;
    // variables
    double cosphi, sinphi;
    double val0, val1, val2, val3;
    double w0, w1, w2;
    double xc, zc, xb, xv;
    float ct;
    sq_params p;

    ind_a = 0;
    for (ia = 0; ia < nangles; ia++) {

        // make_sq_lut(sq_lut, angles[ia]);

        cosphi = cos(angles[ia]);
        sinphi = sin(angles[ia]);
        p = sqpoints(angles[ia]);

        nnz = 0;
        for (i = 0; i < nrays; i++) {
            xc = cord(0, i) - xcent;
            zc = cord(1, i) - zcent;
            xb = xc * cosphi + zc * sinphi + xcent;
            // take neighbouring weights /pm 1 pixel
            // iqx = (int)roundf(xb);
            iqx = double2int(xb); // this is much faster, make check to see if valid

            xv = iqx - xb;
            val0 = piece_wise_integrated(xv - 1.5, p.xmin, p.xmax, p.lmax);
            val1 = piece_wise_integrated(xv - 0.5, p.xmin, p.xmax, p.lmax);
            val2 = piece_wise_integrated(xv + 0.5, p.xmin, p.xmax, p.lmax);
            val3 = piece_wise_integrated(xv + 1.5, p.xmin, p.xmax, p.lmax);

            w0 = val1 - val0;
            w1 = val2 - val1;
            w2 = val3 - val2;

            ind_y = ind_a;
            for (iqy = 0; iqy < ny; iqy++) {
                ct = x[nnz + iqy];

                if (iqx > 0 && iqx + 1 < nx) {
                    y[ind_y + (iqx - 1)] += w0 * ct;
                    y[ind_y + (iqx)] += w1 * ct;
                    y[ind_y + (iqx + 1)] += w2 * ct;
                } else {
                    if (is_inside(iqx - 1, nx)) { y[ind_y + (iqx - 1)] += w0 * ct; }
                    if (is_inside(iqx, nx)) { y[ind_y + (iqx)] += w1 * ct; }
                    if (is_inside(iqx + 1, nx)) { y[ind_y + (iqx + 1)] += w2 * ct; }
                }
                ind_y += nx;
            } /* end for iqx */
            nnz = nnz + ny;
        } /* end for i */
        if (nnz0 != nnz) {
            fprintf(stderr, "arecProject2D: problem with nnz count!\n");
            status = 2;
            break;
        }
        ind_a += nx * ny;
    } /* end for ia */

    return status;
}

int arecBackProject2D_SQ(arecImage const images, float *angles, int nangles, arecImage *bvol) {

    int status = 0;

    int const nx = images.nx;
    int const ny = images.ny;
    int const nz = images.nz;

    if (nz != nangles) {
        status = 1;
        return status;
    }
    float *x = images.data;

    if (bvol->nx != nx || bvol->ny != ny) {
        status = -1;
        fprintf(stderr, "arecBackProject2D: image and volume sizes don't match\n");
        return status;
    }

    double const radius = bvol->radius;
    int const nrays = bvol->nrays;
    int const nnz0 = bvol->nnz;
    int *cord = bvol->cord;

    if (!bvol->is_cyl) {
        status = -1;
        fprintf(stderr, "arecBackProject2D: cylindrical volume ");
        fprintf(stderr, "not properly set!\n");
        return status;
    }
    float *y = bvol->data;
    reset(bvol);

    double const xcent = radius;
    double const zcent = radius;

    // loops
    int ia, i, j, iqx;
    int nnz, ind_a, ind_y;
    // variables
    double cosphi, sinphi;
    double val0, val1, val2, val3;
    double w0, w1, w2;
    double xc, zc, xb, xv;
    // sq_params p;

    ind_a = 0;
    for (ia = 0; ia < nangles; ia++) {
        cosphi = cos(angles[ia]);
        sinphi = sin(angles[ia]);
        sq_params p = sqpoints(angles[ia]);
        nnz = 0;
        for (i = 0; i < nrays; i++) {
            xc = cord(0, i) - xcent;
            zc = cord(1, i) - zcent;
            xb = xc * cosphi + zc * sinphi + xcent;

            // take neighbouring weights /pm 1 pixel
            // iqx = (int)roundf(xb);
            iqx = double2int(xb); // this is much faster, make check to see if valid
            xv = iqx - xb;
            val0 = piece_wise_integrated(xv - 1.5, p.xmin, p.xmax, p.lmax);
            val1 = piece_wise_integrated(xv - 0.5, p.xmin, p.xmax, p.lmax);
            val2 = piece_wise_integrated(xv + 0.5, p.xmin, p.xmax, p.lmax);
            val3 = piece_wise_integrated(xv + 1.5, p.xmin, p.xmax, p.lmax);

            w0 = val1 - val0;
            w1 = val2 - val1;
            w2 = val3 - val2;

            if (iqx > 0 && iqx + 1 < nx) { // all possible pixles inside
                ind_y = ind_a;
                for (j = 0; j < ny; j++) {
                    y[nnz + j] += w0 * x[iqx - 1 + ind_y];
                    y[nnz + j] += w1 * x[iqx + ind_y];
                    y[nnz + j] += w2 * x[iqx + 1 + ind_y];
                    ind_y += nx;
                }
            } else { // check boundaries separately
                if (is_inside(iqx - 1, nx)) {
                    ind_y = ind_a;
                    for (j = 0; j < ny; j++) {
                        y[nnz + j] += w0 * x[iqx - 1 + ind_y];
                        ind_y += nx;
                    }
                }
                if (is_inside(iqx, nx)) {
                    ind_y = ind_a;
                    for (j = 0; j < ny; j++) {

                        y[nnz + j] += w1 * x[iqx + ind_y];
                        ind_y += nx;
                    }
                }
                if (is_inside(iqx + 1, nx)) {
                    ind_y = ind_a;
                    for (j = 0; j < ny; j++) {
                        y[nnz + j] += w2 * x[iqx + 1 + ind_y];
                        ind_y += nx;
                    }
                }
            }

            nnz = nnz + ny;
        } /* end for i */
        if (nnz0 != nnz) {
            fprintf(stderr, "arecProject2D: problem with nnz count!\n");
            status = 2;
            break;
        }
        ind_a += nx * ny;
    } /* end for ia */

    return status;
}
