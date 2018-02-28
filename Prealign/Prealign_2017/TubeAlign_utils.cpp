#include "TubeAlign_utils.h"
#include "NCXT-image\image2d.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

typedef std::vector<image2d<float>> image_stack;

int findMaxIndex(const float *data, size_t N) {
    float max_val = data[0];
    int max_ind = 0;
    for (int i = 0; i < N; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_ind = i;
        }
    }
    return max_ind;
}

// SLOPE FIT
/* Simple routine to print the fit results */
void printresult(double const *x, mp_result *result) {
    if ((x == 0) || (result == 0)) return;
    printf("  CHI-SQUARE = %f    (%d DOF)\n", result->bestnorm, result->nfunc - result->nfree);
    printf("        NPAR = %d\n", result->npar);
    printf("       NFREE = %d\n", result->nfree);
    printf("     NPEGGED = %d\n", result->npegged);
    printf("     NITER = %d\n", result->niter);
    printf("      NFEV = %d\n", result->nfev);
    printf("\n");
    {
        for (int i = 0; i < result->npar; i++) {
            printf("  P[%d] = %f +/- %f\n", i, x[i], result->xerror[i]);
        }
    }
}
/*
 * rotation fit function
 *
 * m - number of data points
 * n - number of parameters (3)
 * p - array of fit parameters
 *     p[0] = phase shift
 *     p[1] = tilt angle
 *     p[2] = camera rotation
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int rotationfunc(int m, int n, double *p, double *dy, double **dvec, void *vars) {
    int i;
    struct vars_struct *v = (struct vars_struct *)vars;
    double *x, *y, *ey;

    x = v->x;
    y = v->y;
    ey = v->ey;
    for (i = 0; i < m; i++) {
        double y_val = atan(cos(x[i] - p[0]) * tan(p[1])) + p[2];
        dy[i] = (y[i] - y_val) / ey[i];
    }
    return 0;
}

int fitRotation(std::vector<double> &x, std::vector<double> &y, std::vector<double> &ey,
                std::vector<double> &p, double &norm, bool verb) {
    const int npar = 3;
    int N = x.size();
    assert(y.size() == N);
    assert(ey.size() == N);
    assert(p.size() == npar);

    double perror[npar]; /* Returned parameter errors */
    mp_par pars[npar];   /* Parameter constraints */
                         // int i;
    struct vars_struct v;
    int status;
    mp_result result;

    memset(&result, 0, sizeof(result)); /* Zero results structure */
    result.xerror = perror;

    memset(pars, 0, sizeof(pars)); /* Initialize constraint structure */

    v.x = x.data();
    v.y = y.data();
    v.ey = ey.data();
    status = mpfit(rotationfunc, N, npar, p.data(), pars, 0, (void *)&v, &result);

    if (verb) printf("*** testRotation status = %d\n", status);
    if (verb) printresult(p.data(), &result);
    norm = result.bestnorm;
    return 0;
}

void linFitEdgeWerr(const int *data, int N, double &p1, double &p2, double &p1_err) {
    double EPS = 0.001;
    double sumx = 0.0, sumy = 0.0;
    int n_vals = 0;
    double cov = 0.0, var = 0.0, err_sum = 0.0;

    // calculate mean of non-negative edges
    for (int i = 1; i < N; i++) {
        if (data[i] > 0) {
            sumx += 1.0 * i;
            sumy += data[i];
            n_vals++;
        }
    }
    double meanx = 1.0 * sumx / n_vals;
    double meany = 1.0 * sumy / n_vals;

    // Calculate covariance and variance
    for (int i = 1; i < N; i++) {
        if (data[i] > 0) {
            cov += (1.0 * i - meanx) * (data[i] - meany);
            var += (1.0 * i - meanx) * (1.0 * i - meanx);
        }
    }

    p1 = cov / var;
    p2 = meany - meanx * p1;

    for (int i = 1; i < N; i++) {
        if (data[i] > 0) { err_sum += (data[i] - (p1 * i + p2)) * (data[i] - (p1 * i + p2)); }
    }

    double S = sqrt(EPS + err_sum / (n_vals - 2.0)); // EPS+ to ensure nonzero error;
    p1_err = S * sqrt(n_vals / var);
    // printf("(%4.2f,%4.2f) %4.5f   \n",p1,p2, p1_err);
}
void ransac(const int *Y, int size_Y, int iter, double TH, double &p1, double &p2) {
    int bestNUM = 0;
    double pc1 = 0.0;
    double pc2 = 0.0;

    std::vector<double> d = std::vector<double>(size_Y);

    for (int i = 0; i < iter; i++) {
        int index1 = rand() % size_Y;
        int index2;
        while ((index2 = rand() % size_Y) == index1)
            ;
        // Compute the distances between all points with the fitting line
        double kX = index2 - index1;
        double kY = Y[index2] - Y[index1];
        double kNorm = sqrt(kX * kX + kY * kY);
        double nVX = -kY / kNorm;
        double nVY = kX / kNorm;
        // calculate distances
        for (int j = 0; j < size_Y; j++)
            d[j] = nVX * (j - index1) + nVY * (Y[j] - Y[index1]);
        // count inliners
        int n_inliner = 0;
        for (int j = 0; j < size_Y; j++)
            if (Y[j] > 0 && abs(d[j]) < TH) n_inliner++;
        // update
        if (n_inliner > bestNUM) {
            bestNUM = n_inliner;
            pc1 = kY / kX;
            pc2 = Y[index1] - index1 * pc1;
        }
    }
    p1 = pc1;
    p2 = pc2;
}
void cleanEdge(int *Y, int size_Y, double TH, double p1, double p2) {
    // Edges that are over TH from line are marked with -1
    double nVX = -p1;
    double nVY = 1.0;
    for (int j = 0; j < size_Y; j++)
        if (abs(nVX * (j) + nVY * (Y[j] - p2)) > TH) Y[j] = -1;
}

// IMAGE OPERATORS

#define B(i) B[(i)-1]
void PRB1D(double const *B, int NPOINT, double *POS) {
    /* array B(NPOINT) */
    double C2 = 0.0;
    double C3 = 0.0;
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
void ccorr1d(int length, float *x, float *y, float *c) {
    int lsam = (length / 2) + 1;

    fftwf_plan px, py, pc;

    fftwf_complex *fx =
        reinterpret_cast<fftwf_complex *>(fftwf_malloc(lsam * sizeof(fftwf_complex)));
    fftwf_complex *fy =
        reinterpret_cast<fftwf_complex *>(fftwf_malloc(lsam * sizeof(fftwf_complex)));
    fftwf_complex *ff =
        reinterpret_cast<fftwf_complex *>(fftwf_malloc(lsam * sizeof(fftwf_complex)));

    /* FFT the image x */
    px = fftwf_plan_dft_r2c_1d(length, x, fx, FFTW_ESTIMATE);
    fftwf_execute(px);
    /* FFT the image y */
    py = fftwf_plan_dft_r2c_1d(length, y, fy, FFTW_ESTIMATE);
    fftwf_execute(py);
    /* conj(fx).*fy */
    for (int i = 0; i < lsam; i++) {
        ff[i][0] = fx[i][0] * fy[i][0] + fx[i][1] * fy[i][1];
        ff[i][1] = fx[i][0] * fy[i][1] - fx[i][1] * fy[i][0];
    }
    pc = fftwf_plan_dft_c2r_1d(length, ff, c, FFTW_ESTIMATE);
    fftwf_execute(pc);

    fftwf_destroy_plan(px);
    fftwf_destroy_plan(py);
    fftwf_destroy_plan(pc);

    fftwf_free(fx);
    fftwf_free(fy);
    fftwf_free(ff);
}

// READ / WRITE
std::vector<double> loadAngles(std::string filename) {
    const double PI = 3.14159265359;
    std::ifstream myfile;
    myfile.open(filename.c_str(), std::ios::in);
    if (!myfile) {
        std::cout << "Failed to load! \n";
        return std::vector<double>();
    }

    std::vector<double> angles = std::vector<double>();
    std::string line;
    while (std::getline(myfile, line)) {
        float angle;
        std::istringstream iss(line);

        if (!(iss >> angle)) {
            // empty line, do nothing
        } else
            angles.push_back(angle * PI / 180); // to rad
    }
    myfile.close();
    std::cout << "loadAngles: Loaded " << angles.size() << " angles from ";
    std::cout << angles[0] * 180 / PI << " to " << angles.back() * 180 / PI << '\n';
    return angles;
}
void write_raw_pnm(const char *fname, char const *pixels, int w, int h) {
    FILE *f;
    errno_t err = fopen_s(&f, fname, "wb");
    // f = fopen(&f, fname, "wb");
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
void saveProfile(image2d<float> img, char const *fname) {
    int Z = img.ny();
    int Y = img.nx();
    std::vector<char> pixels = std::vector<char>(Z * Y * 3);
    // get vals for scaling
    double a = getMinValue(img.data(), Z * Y);
    double b = getMaxValue(img.data(), Z * Y);
    for (int iz = 0; iz < Z; iz++) {
        for (int yi = 0; yi < Y; yi++) {
            double value = img[Y * iz + yi];
            // Profile is saved in reverse order (horizontal Z)
            pixels[3 * (iz + Z * yi) + 0] = static_cast<int>((255 * (value - a) / (b - a)));
            pixels[3 * (iz + Z * yi) + 1] = static_cast<int>((255 * (value - a) / (b - a)));
            pixels[3 * (iz + Z * yi) + 2] = static_cast<int>((255 * (value - a) / (b - a)));
        }
    }

    write_raw_pnm(fname, pixels.data(), Z, Y);
}

void writeInput_quick(const char *dataname, const char *anglename) {
    std::string recname = std::string(dataname);
    size_t lastindex = recname.find_last_of(".");
    if (lastindex != std::string::npos) { recname = recname.substr(0, lastindex); }

    ofstream myfile;
    myfile.open("input_quick");
    myfile << "data     " << dataname << "\n";
    myfile << "angles   " << anglename << "\n";
    // myfile << "cropx    " << image_original.nx-4*PAD_TUBE_EDGE << "\n";
    myfile << "lam      5e-7"
           << "\n";
    myfile << "tolsirt  1e-3"
           << "\n";
    myfile << "maxsirt  5"
           << "\n";
    myfile << "maxiter  0"
           << "\n";
    myfile << "output   testrec.mrc\n";
    myfile << "fudge    40"
           << "\n";
    myfile << "rmethod  2"
           << "\n";
    myfile << "pmethod  3"
           << "\n";
    myfile << "thresh   1.05"
           << "\n";
    myfile.close();
}
void writeInput(const char *dataname, const char *anglename) {
    std::string recname = std::string(dataname);
    size_t lastindex = recname.find_last_of(".");
    if (lastindex != std::string::npos) { recname = recname.substr(0, lastindex); }

    ofstream myfile;
    myfile.open("input");
    myfile << "data     " << dataname << "\n";
    myfile << "angles   " << anglename << "\n";
    // myfile << "cropx    " << image_original.nx-4*PAD_TUBE_EDGE << "\n";
    myfile << "lam      5e-7"
           << "\n";
    myfile << "tolsirt  1e-3"
           << "\n";
    myfile << "maxsirt  3"
           << "\n";
    myfile << "maxiter  8"
           << "\n";
    myfile << "output   " << recname << "_rec.mrc\n";
    myfile << "fudge    40"
           << "\n";
    myfile << "rmethod  2"
           << "\n";
    myfile << "pmethod  3"
           << "\n";
    myfile << "thresh   1.05"
           << "\n";
    myfile.close();
}

void writeAngles(const char *anglename, vector<double> angles) {
    // make 0 0
    const double PI = 3.14159265359;
    double const angle0 = angles[0];
    for (int i = 0; i < angles.size(); i++) {
        // printf("%d %4.2f -> %4.2f \n", i ,angles[i]*180/PI,
        // fmod(angles[i]-angle0+4*PI,2*PI)*180/PI); angles[i] = fmod(angles[i] - angle0 + 4.0 *
        // M_PI, 2.0 * M_PI);
    }

    ofstream myfile;
    myfile.open(anglename);
    for (int i = 0; i < angles.size(); i++) {
        myfile << angles[i] * 180 / PI << "\n";
    }
    myfile.close();
}
void writePrealignment(const char *file, double rot, vector<int> sx, vector<int> sx2,
                       vector<int> sy) {
    ofstream myfile;
    stringstream filename;
    filename << file;
    myfile.open(filename.str().c_str());

    for (int i = 0; i < sx.size(); i++) {
        myfile << rot << " " << sx[i] + sx2[i] << " " << sy[i] << "\n";
    }
    myfile.close();
}
void writeArray(double const *data, char *file, int N) {
    ofstream myfile;
    stringstream filename;
    filename << file;
    myfile.open(filename.str().c_str());
    for (int i = 0; i < N; i++) {
        myfile << data[i] << "\n";
    }
    myfile.close();
}

// PROFILE ALIGN
float getLineMean(image_stack &img, int z, int y, int x_min, int x_max) {
    int nx = img[0].nx();
    int ny = img[0].ny();

    double sum = 0;
    for (int xi = x_min; xi < x_max; xi++)
        sum += img[z].m_data[nx * y + xi];
    return static_cast<float>(sum / (x_max - x_min));
}
void circShiftX(float *data, int length, int shift) {
    std::vector<float> buffer = std::vector<float>(abs(shift));
    /*
    The three loops are
    1: Copy periodically shifted data
    2: Shift remaining data in-place
    3; Put back the copied data into array
    */
    if (shift > 0) {
        for (int i = 0; i < shift; i++)
            buffer[i] = data[length - shift + i];
        for (int i = length - 1; i >= shift; i--)
            data[i] = data[i - shift];
        for (int i = 0; i < shift; i++)
            data[i] = buffer[i];
    } else if (shift < 0) {
        for (int i = 0; i < -shift; i++)
            buffer[i] = data[i];
        for (int i = -shift; i < length; i++)
            data[i + shift] = data[i];
        for (int i = 0; i < -shift; i++)
            data[length + shift + i] = buffer[i];
    }

    return;
}
double findMaxCC_fit(float const *p_cccoefs, int length) {
    double pos, ret;
    double plist[7];
    int sYI = findMaxIndex(p_cccoefs, length);
    /* fit a polynomial around the peak, and return the position of
    the maximum of the polynomial */
    for (int k = -3; k <= 3; k++) {
        plist[k + 3] = p_cccoefs[(sYI + k + length) % length];
    }
    PRB1D(plist, 7, &pos); /* fit with a polynomial and find its maximizer */

    ret = 1.0 * (sYI + pos);
    if (ret > length / 2) ret -= length;
    return ret;
}

// Convolutions
vector<float> getGaussianDerivateKernel(double n_sigma, double sigma) {
    const double PI = 3.14159265359;

    int halfkernel = static_cast<int>(std::floor(n_sigma * sigma));
    size_t kernel_size = 2 * halfkernel + 1;
    vector<float> Y(kernel_size, 0.0);

    for (int i = 0; i < kernel_size; i++) {
        double x = (1.0 * i - halfkernel);
        Y[i] = static_cast<float>(-x * 1.0 / (sigma * sigma * sigma * sqrt(2.0 * PI)) *
                                  exp(-x * x / (2.0 * sigma * sigma)));
    }

    return Y;
}
vector<float> convolve_valid(float *data, const float *kernel, float fill_val, int len,
                             int len_halfkernel) {
    vector<float> buffer(len, fill_val);

    // convolve
    for (int i = len_halfkernel; i < len - len_halfkernel; i++) {
        for (int m = -len_halfkernel; m <= len_halfkernel; m++) {
            buffer[i] += data[i - m] * kernel[m + len_halfkernel];
        }
    }
    return buffer;
}
void convolve_valid_inplace(float *data, float *kernel, float fill_val, int len,
                            int len_halfkernel) {
    vector<float> buffer(len, fill_val);

    // convolve
    for (int i = len_halfkernel; i < len - len_halfkernel; i++) {
        for (int m = -len_halfkernel; m <= len_halfkernel; m++) {
            buffer[i] += data[i - m] * kernel[m + len_halfkernel];
        }
    }

    // copy back
    for (int i = 0; i < len; i++)
        data[i] = buffer[i];
}
