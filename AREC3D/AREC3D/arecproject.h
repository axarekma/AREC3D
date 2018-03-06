#pragma once
#include <algorithm>
#include <math.h>
#include <stdio.h>

// new weight params
typedef struct {
    double xmin, xmax, lmax;
} sq_params;

inline sq_params sqpoints(double angle) {
    double cosphi = cos(angle);
    double sinphi = sin(angle);
    double cms = fabs(0.5 * (cosphi - sinphi));
    double cps = fabs(0.5 * (cosphi + sinphi));

    double a = std::min(cms, cps);
    double b = std::max(cms, cps) + 1e-6;
    double lmax = 1 / (((b - a) + 2 * a));
    sq_params retval = {a, b, lmax};
    return retval;
}

inline double piece_wise_integrated(double const x, const double a, const double b,
                                    const double y_max) {
    if (x < -b) { return 0.0; }
    if (x > b) { return 1.0; }

    auto f1 = [a, b, y_max](double arg) -> double {
        return 0.5 * y_max * (arg + b) * (arg + b) / (b - a);
    };
    auto f2 = [a, b, y_max](double arg) -> double { return 0.5 * y_max * (b + a) + y_max * arg; };
    auto f3 = [a, b, y_max](double arg) -> double {
        return 0.5 * y_max * (b - arg) * (b - arg) / (a - b) + 1.0;
    };
    const double a_eps = a + 1e-3;

    // printf("f1 = %4.2f f2 = %4.2f f3 = %4.2f\n", f1(x), f2(x), f3(x));

    if (x > -a_eps && x < a_eps) { return f2(x); }
    if (x < -a) { return f1(x); }
    if (x > a) { return f3(x); }

    // no return value, something wrong
    printf("No case found, this should not happen\n");
    return 0.0;
}

inline bool is_inside(int x, int nx) {
    if (x < 0) { return false; }
    if (x > (nx - 1)) { return false; }
    return true;
}

int arecBackProject2D(arecImage images, float *angles, int nangles, arecImage *bvol);
int arecProject2D(arecImage cylvol, float *angles, int nangles, arecImage *projstack);

int arecBackProject2D_KB(arecImage images, float *angles, int nangles, arecImage *bvol);
int arecProject2D_KB(arecImage cylvol, float *angles, int nangles, arecImage *projstack);

int arecBackProject2D_SQ(arecImage images, float *angles, int nangles, arecImage *bvol);
int arecProject2D_SQ(arecImage cylvol, float *angles, int nangles, arecImage *projstack);
