#include "fftw3.h"
#include <algorithm>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include "arecConstants.h"
#include "arecImage.h"
#include "imagetools.h"

/*
Doing a periodic decomposition of the image:
Moisan, Lionel.
"Periodic plus smooth image decomposition."
Journal of Mathematical Imaging and Vision 39.2 (2011): 161-179.
*/
void makePeriodicSlice(int nx, int ny, float *data, float *out) {
    int i, j;
    int ix, iy;
    float *v;
    fftwf_plan px, pc;
    fftwf_complex *cx, *ff;

    v = (float *)calloc(nx * ny, sizeof(float));
    for (i = 0; i < nx; i++) {
        v[0 * nx + i] = data[nx * (ny - 1) + i] - data[nx * (0) + i];
        v[(ny - 1) * nx + i] = data[nx * (0) + i] - data[nx * (ny - 1) + i];
    }
    for (i = 0; i < ny; i++) {
        v[i * nx + 0] += data[nx * i + (nx - 1)] - data[nx * i + 0];
        v[i * nx + (nx - 1)] += data[nx * i + 0] - data[nx * i + (nx - 1)];
    }

    /* put V in a complex array */
    cx = (fftwf_complex *)fftwf_malloc(ny * nx * sizeof(fftwf_complex));
    for (i = 0; i < nx * ny; i++) {
        cx[i][0] = v[i];
        cx[i][1] = 0.0;
    }

    /* FFT V */
    ff = (fftwf_complex *)fftwf_malloc(ny * nx * sizeof(fftwf_complex));
    px = fftwf_plan_dft_2d(ny, nx, cx, ff, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(px);

    // calculate S
    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
            ff[nx * j + i][0] /= (2 * cos(2.0 * PI * i / nx) + 2 * cos(2.0 * PI * j / ny) - 4);
            ff[nx * j + i][1] /= (2 * cos(2.0 * PI * i / nx) + 2 * cos(2.0 * PI * j / ny) - 4);
        }
    }
    ff[0][0] = 0.0;
    ff[0][1] = 0.0;

    // inverse S
    pc = fftwf_plan_dft_2d(ny, nx, ff, cx, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(pc);

    for (i = 0; i < nx * ny; i++) {
        cx[i][0] /= (float)(nx * ny);
    }

    // Exporting the smooth part (Image=P+S)
    for (i = 0; i < nx * ny; i++) {
        out[i] = cx[i][0];
    }

    free(v);
    fftwf_destroy_plan(px);
    fftwf_destroy_plan(pc);
    fftwf_free(cx);
    fftwf_free(ff);
}

/*
Do a periodic decomposition of the image U=P+S
The final image is the obtained by I=U-W*S,
where W is a Gaussian weight function around the border
large sigma can introduce alterations on the original data
but it seems a smoothing of sigma<10 pix is enough.
*/
#define SIGMA_BORDER 5.0 // width of the border window used for periodic smoothing
void periodicDecomposition(arecImage img) {
    int zi, i;
    float *data;
    float *S;
    int nx = img.nx;
    int ny = img.ny;
    int nz = img.nz;
    data = img.data;

    S = (float *)malloc(nx * ny * sizeof(float));
    for (zi = 0; zi < nz; zi++) {
        // float* S=new float[nx*ny];
        makePeriodicSlice(nx, ny, &data[nx * ny * zi], S);
        for (i = 0; i < nx * ny; i++) {
            // d= distance to nearest border.
            int iy = i / nx;
            int ix = i - (i / nx) * nx;
            double d =
                1.0 * std::min(std::min(ix, abs(nx - 1 - ix)), std::min(iy, abs(ny - 1 - iy)));
            // Apply gaussian window on smoother to preserve data
            // Original decomposition has potentially large gradients in S.
            double sigma = 2.0;
            double W = exp(-1.0 * d * d / (2 * SIGMA_BORDER * SIGMA_BORDER));
            data[nx * ny * zi + i] -= W * S[i];
        }
        // delete [] S;
    }
    free(S);
}

#define depth 5            // depth of smoothing
#define n_smooth 2 * 2 + 1 // width of kernel ODD!
#define halfW depth + (n_smooth - 1) / 2
void periodicSmoothing(arecImage img) {
    int nx, ny, nimages;
    int x, y, z, i, j;
    float *data;
    nx = img.nx;
    ny = img.ny;
    nimages = img.nz;
    data = img.data;
    for (z = 0; z < nimages; z++) {
        // horizontal border
        for (x = 0; x < nx; x++) {
            // fill array with image
            float vals[depth * 2 + n_smooth - 1];
            for (i = 0; i < (halfW); i++) {
                vals[i] = data[nx * ny * z + nx * (ny - (halfW - i)) + x];
                vals[halfW + i] = data[nx * ny * z + nx * (i) + x];
            }
            // update image with mean
            for (i = 0; i < depth; i++) {
                data[nx * ny * z + nx * (ny - (depth - i)) + x] *= 0.0;
                data[nx * ny * z + nx * (i) + x] *= 0.0;
                for (j = 0; j < n_smooth; j++) {
                    data[nx * ny * z + nx * (ny - (depth - i)) + x] += vals[i + j];
                    data[nx * ny * z + nx * (i) + x] += vals[depth + i + j];
                }
                data[nx * ny * z + nx * (ny - (depth - i)) + x] /= n_smooth;
                data[nx * ny * z + nx * (i) + x] /= n_smooth;
            }
        }
        // vertical border
        for (y = 0; y < ny; y++) {
            // fill array with image
            float vals[depth * 2 + n_smooth - 1];
            for (i = 0; i < (halfW); i++) {
                vals[i] = data[nx * ny * z + nx * y + (nx - (halfW - i))];
                vals[halfW + i] = data[nx * ny * z + nx * y + (i)];
            }
            // update image with mean
            for (i = 0; i < depth; i++) {
                data[nx * ny * z + nx * y + (nx - (depth - i))] *= 0.0;
                data[nx * ny * z + nx * y + (i)] *= 0.0;
                for (j = 0; j < n_smooth; j++) {
                    data[nx * ny * z + nx * y + (nx - (depth - i))] += vals[i + j];
                    data[nx * ny * z + nx * y + (i)] += vals[depth + i + j];
                }
                data[nx * ny * z + nx * y + (nx - (depth - i))] /= n_smooth;
                data[nx * ny * z + nx * y + (i)] /= n_smooth;
            }
        }
    }
}

void circShiftX(float *data, int length, int shift) {
    int i;
    int bufflen = abs(shift);
    /*
    The three loops are
    1: Copy periodically shifted data
    2: Shift remaining data in-place
    3; Put back the copied data into array
    */
    float *buffer = (float *)malloc(bufflen * sizeof(float));
    if (shift > 0) {
        for (i = 0; i < shift; i++)
            buffer[i] = data[length - shift + i];
        for (i = length - 1; i >= shift; i--)
            data[i] = data[i - shift];
        for (i = 0; i < shift; i++)
            data[i] = buffer[i];
    } else if (shift < 0) {
        for (i = 0; i < -shift; i++)
            buffer[i] = data[i];
        for (i = -shift; i < length; i++)
            data[i + shift] = data[i];
        for (i = 0; i < -shift; i++)
            data[length + shift + i] = buffer[i];
    }
    free(buffer);
    return;
}

void circShiftY(float *data, int nx, int length, int shift, int x) {
    int i;
    int bufflen = abs(shift);
    /*
    The three loops are
    1: Copy periodically shifted data
    2: Shift remaining data in-place
    3; Put back the copied data into array
    */
    float *buffer = (float *)malloc(bufflen * sizeof(float));
    if (shift > 0) {
        for (i = 0; i < shift; i++)
            buffer[i] = data[(length - shift + i) * nx + x];
        for (i = length - 1; i >= shift; i--)
            data[i * nx + x] = data[(i - shift) * nx + x];
        for (i = 0; i < shift; i++)
            data[i * nx + x] = buffer[i];
    } else if (shift < 0) {
        for (i = 0; i < -shift; i++)
            buffer[i] = data[i * nx + x];
        for (i = -shift; i < length; i++)
            data[(i + shift) * nx + x] = data[i * nx + x];
        for (i = 0; i < -shift; i++)
            data[(length + shift + i) * nx + x] = buffer[i];
    }
    free(buffer);
    return;
}

void circShift2D(float *data, int nx, int ny, int shift, int index, int dim) {
    if (dim == 0) {
        /*Horizontal shift:
        shift row 'index' by 'shift' pixels
        */
        circShiftX(&data[index * nx], nx, shift);
    } else if (dim == 1) {
        /*Vertical shift:
        shift column 'index' by 'shift' pixels
        */
        circShiftY(data, nx, ny, shift, index);
    }
    return;
}

void shearX_circ(float *img, float shear, int nx, int ny) {
    double center = 0.5 * (ny - 1);
    for (int y = 0; y < ny; y++) {
        float skew = shear * (y - center);
        int skewi = floor(skew);
        float skewf = skew - 1.0 * skewi;
        float oleft = 0.0;

        circShift2D(img, nx, ny, skewi, y, 0);

        float pixel0 = img[y * nx];
        for (int x = 0; x < nx; x++) {
            float pixel = img[y * nx + x];
            float left = pixel * skewf;
            pixel = pixel - left + oleft;
            img[y * nx + x] = pixel;
            oleft = left;
        }
        // Fix first pixel
        float left = pixel0 * skewf;
        float pixel = (pixel0 - left) + oleft;
        img[y * nx] = pixel;
    }
}
void shearY_circ(float *img, float shear, int nx, int ny) {
    double center = 0.5 * (nx - 1);
    for (int x = 0; x < nx; x++) {
        float skew = shear * (x - center);
        int skewi = floor(skew);
        float skewf = skew - 1.0 * skewi;
        float oleft = 0.0;

        circShift2D(img, nx, ny, skewi, x, 1);

        float pixel0 = img[x];
        for (int y = 0; y < ny; y++) {
            float pixel = img[y * nx + x];
            float left = pixel * skewf;
            pixel = pixel - left + oleft;
            img[y * nx + x] = pixel;
            oleft = left;
        }
        // Fix first pixel
        float left = pixel0 * skewf;
        float pixel = (pixel0 - left) + oleft;
        img[x] = pixel;
    }
}

void blackmanHarris_filter(float *in, float *out, int nx, int ny) {
    double a0 = 0.35875;
    double a1 = 0.48829;
    double a2 = 0.14128;
    double a3 = 0.01168;
    int x, y;

    for (y = 0; y < ny; y++) {
        double by = 2.0 * M_PI * y / (ny - 1);
        double hy = a0 - a1 * cos(by) + a2 * cos(2 * by) - a3 * cos(3 * by);
        for (x = 0; x < nx; x++) {
            double bx = 2.0 * M_PI * x / (nx - 1);
            double hx = a0 - a1 * cos(bx) + a2 * cos(2 * bx) - a3 * cos(3 * bx);

            out[y * nx + x] = in[y * nx + x] * hx * hy;
        }
    }
}
void blackmanHarris_filter_normalize(float *in, float *out, int nx, int ny) {
    double a0 = 0.35875;
    double a1 = 0.48829;
    double a2 = 0.14128;
    double a3 = 0.01168;
    int x, y;

    float meanval = getArrayMean(in, nx * ny);

    for (y = 0; y < ny; y++) {
        double by = 2.0 * M_PI * y / (ny - 1);
        double hy = a0 - a1 * cos(by) + a2 * cos(2 * by) - a3 * cos(3 * by);
        for (x = 0; x < nx; x++) {
            double bx = 2.0 * M_PI * x / (nx - 1);
            double hx = a0 - a1 * cos(bx) + a2 * cos(2 * bx) - a3 * cos(3 * bx);

            out[y * nx + x] = (in[y * nx + x] - meanval) * hx * hy;
        }
    }
}

void tukey_filter(float *in, float *out, int nx, int ny, double alpha) {
    double xa = (0.5 * alpha * (nx - 1));
    double xb = (1.0 - 0.5 * alpha) * (nx - 1);
    double ya = (0.5 * alpha * (ny - 1));
    double yb = (1.0 - 0.5 * alpha) * (ny - 1);
    double A = 2.0 * M_PI / alpha;

    int x, y;
    for (y = 0; y < ny; y++) {
        double hy;
        double valya = 0.5 * (1.0 + cos(A * (1.0 * y / (ny - 1) - alpha / 2)));
        double valyb = 0.5 * (1.0 + cos(A * (1.0 * y / (ny - 1) - 1.0 + alpha / 2)));

        if (y < ya)
            hy = valya;
        else if (y > yb)
            hy = valyb;
        else
            hy = 1.0;

        for (x = 0; x < nx; x++) {
            double hx;
            double valxa = 0.5 * (1.0 + cos(A * (1.0 * x / (nx - 1) - alpha / 2)));
            double valxb = 0.5 * (1.0 + cos(A * (1.0 * x / (nx - 1) - 1.0 + alpha / 2)));

            if (x < xa)
                hx = valxa;
            else if (x > xb)
                hx = valxb;
            else
                hx = 1.0;

            out[y * nx + x] = in[y * nx + x] * hx * hy;
        }
    }
}

void tukey_filter2d_inplace(float *in, int nx, int ny, double alpha) {
    double xa = (0.5 * alpha * (nx - 1));
    double xb = (1.0 - 0.5 * alpha) * (nx - 1);
    double ya = (0.5 * alpha * (ny - 1));
    double yb = (1.0 - 0.5 * alpha) * (ny - 1);
    double A = 2.0 * M_PI / alpha;

    int x, y;
    for (y = 0; y < ny; y++) {
        double hy;
        double valya = 0.5 * (1.0 + cos(A * (1.0 * y / (ny - 1) - alpha / 2)));
        double valyb = 0.5 * (1.0 + cos(A * (1.0 * y / (ny - 1) - 1.0 + alpha / 2)));

        if (y < ya)
            hy = valya;
        else if (y > yb)
            hy = valyb;
        else
            hy = 1.0;

        for (x = 0; x < nx; x++) {
            double hx;
            double valxa = 0.5 * (1.0 + cos(A * (1.0 * x / (nx - 1) - alpha / 2)));
            double valxb = 0.5 * (1.0 + cos(A * (1.0 * x / (nx - 1) - 1.0 + alpha / 2)));

            if (x < xa)
                hx = valxa;
            else if (x > xb)
                hx = valxb;
            else
                hx = 1.0;

            in[y * nx + x] *= hx;
        }
    }
}

double getNormalizedCrossCorrelation(float *im1, float *im2, int nx, int ny) {
    double mean_im1 = 0.0;
    double mean_im2 = 0.0;
    double cov = 0.0;
    double var1 = 0.0;
    double var2 = 0.0;
    int i;
    for (i = 0; i < nx * ny; i++) {
        mean_im1 += im1[i];
        mean_im2 += im2[i];
    }
    mean_im1 /= 1.0 * nx * ny;
    mean_im2 /= 1.0 * nx * ny;

    for (i = 0; i < nx * ny; i++) {
        cov += (im1[i] - mean_im1) * (im2[i] - mean_im2);
        var1 += (im1[i] - mean_im1) * (im1[i] - mean_im1);
        var2 += (im2[i] - mean_im2) * (im2[i] - mean_im2);
    }
    return cov / sqrt(var1 * var2);
}

double getNormalizedCrossCorrelationWithFilter(float *im1, float *im2, int nx, int ny) {
    double mean_im1 = 0.0;
    double mean_im2 = 0.0;
    double cov = 0.0;
    double var1 = 0.0;
    double var2 = 0.0;
    int i;
    float *filter;

    // setup filter
    filter = (float *)malloc(nx * ny * sizeof(float));
    for (i = 0; i < nx * ny; i++)
        filter[i] = 1.0;
    tukey_filter(filter, filter, nx, ny, 0.3);
    // blackmanHarris_filter(filter,filter,  nx,  ny);

    for (i = 0; i < nx * ny; i++) {
        mean_im1 += im1[i] * filter[i];
        mean_im2 += im2[i] * filter[i];
    }
    mean_im1 /= 1.0 * nx * ny;
    mean_im2 /= 1.0 * nx * ny;

    for (i = 0; i < nx * ny; i++) {
        cov += (im1[i] * filter[i] - mean_im1) * (im2[i] * filter[i] - mean_im2);
        var1 += (im1[i] * filter[i] - mean_im1) * (im1[i] * filter[i] - mean_im1);
        var2 += (im2[i] * filter[i] - mean_im2) * (im2[i] * filter[i] - mean_im2);
    }

    free(filter);

    return cov / sqrt(var1 * var2);
}

float getArrayMean(float *data, int N) {
    int i;
    float sum = 0;
    for (i = 0; i < N; i++)
        sum += data[i];
    return sum / N;
}
