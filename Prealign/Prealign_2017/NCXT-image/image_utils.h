#pragma once
#include "image2d.h"
#include "image3d.h"
#include <cmath>
#include <iomanip>
#include <iostream>

#define PI 3.141502

/*
cannot define here, multiple definitions
auto img_abs = [](auto x) -> auto { return abs(x); };
auto img_exp = [](auto x) -> auto { return exp(x); };
auto img_log = [](auto x) -> auto { return log(x); };
auto img_pow2 = [](auto x) -> auto { return x*x; };
auto img_abs_t = [](auto& x) { x = abs(x); };
auto img_exp_t = [](auto& x) { x = exp(x); };
auto img_log_t = [](auto& x) { x = log(x); };
auto img_pow2_t = [](auto& x) { x = x*x; };
/**/

struct Coord2D {
    double x;
    double y;
    Coord2D(double a, double b) : x(a), y(b) {}
};
struct Coord3D {
    double x;
    double y;
    double z;
};

class TransformMatrix2D {
  public:
    double a, b, c, d, e, f, g, h, i;

    TransformMatrix2D() : a(1.0), b(0.0), c(0.0), d(0.0), e(1.0), f(0.0), g(0.0), h(0.0), i(1.0){};

    TransformMatrix2D(double aa, double ab, double ac, double ad, double ae, double af, double ag,
                      double ah, double ai)
        : a(aa), b(ab), c(ac), d(ad), e(ae), f(af), g(ag), h(ah), i(ai){};

    TransformMatrix2D prod(const TransformMatrix2D &n) const {
        return TransformMatrix2D(
            a * n.a + b * n.d + c * n.g, a * n.b + b * n.e + c * n.h, a * n.c + b * n.f + c * n.i,
            d * n.a + e * n.d + f * n.g, d * n.b + e * n.e + f * n.h, d * n.c + e * n.f + f * n.i,
            g * n.a + h * n.d + i * n.g, g * n.b + h * n.e + i * n.h, g * n.c + h * n.f + i * n.i);
    }

    Coord2D prod(const Coord2D &coord) {
        return Coord2D(a * coord.x + b * coord.y + c, d * coord.x + e * coord.y + f);
    }
};
inline TransformMatrix2D rotation_matrix_rad(double angle) {
    return TransformMatrix2D(cos(angle), -sin(angle), 0.0, sin(angle), cos(angle), 0.0, 0.0, 0.0,
                             1.0);
}
inline TransformMatrix2D rotation_matrix_deg(double angle) {
    // positive angle is counter-clockwise
    return rotation_matrix_rad(angle * PI / 180.0);
}
inline TransformMatrix2D translation_matrix(double tx, double ty) {
    // Translation defined as moving the image
    // or equivalently shifting the coordinate
    // system with a factor -1
    return TransformMatrix2D(1.0, 0.0, -tx, 0.0, 1.0, -ty, 0.0, 0.0, 1.0);
}
inline std::ostream &operator<<(std::ostream &os, const TransformMatrix2D &M) {
    std::cout << std::fixed << std::setprecision(1);
    os << M.a << ' ' << M.b << ' ' << M.c << '\n';
    os << M.d << ' ' << M.e << ' ' << M.f << '\n';
    os << M.g << ' ' << M.h << ' ' << M.i << '\n';
    return os;
}
inline TransformMatrix2D chain_transormations(const TransformMatrix2D M1,
                                              const TransformMatrix2D M2) {
    return M1.prod(M2);
}
inline void chain_transormations_inplace(TransformMatrix2D &M1, const TransformMatrix2D M2) {
    auto M_new = chain_transormations(M1, M2);
    std::swap(M1, M_new);
}

template <class T> image2d<T> transform_clip(const image2d<T> &img, TransformMatrix2D M) {
    double center_x = 0.5 * (img.nx() - 1);
    double center_y = 0.5 * (img.ny() - 1);
    auto img_out = image2d<T>(img.nx(), img.ny());
    for (size_t j = 0; j < img.ny(); j++) {
        for (size_t i = 0; i < img.nx(); i++) {
            Coord2D c = M.prod(Coord2D(1.0 * i - center_x, 1.0 * j - center_y));
            img_out(i, j) = img.blerp(c.x + center_x, c.y + center_y);
        }
    }
    return img_out;
}

template <class T> void transform_clip_inplace(image2d<T> &img, TransformMatrix2D M) {
    img = (transform_clip(img, M));
}

template <class T> void print_image(const image2d<T> img) {
    std::cout << "2d image of size (" << img.nx() << ", " << img.ny() << ")\n";
    for (int j = 0; j < img.ny(); j++) {
        for (int i = 0; i < img.nx(); i++) {
            std::cout << std::fixed << std::setprecision(0) << img(i, j) << " ";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
    std::cout << '\n';
}

template <class T> image3d<T> combine_stack(std::vector<image2d<T>> imgstack) {
    int nx = imgstack[0].nx();
    int ny = imgstack[0].ny();
    for (auto img : imgstack) {
        assert(img.nx() == nx);
        assert(img.ny() == ny);
    }

    image3d<T> retval(nx, ny, imgstack.size());
    for (int i = 0; i < imgstack.size(); i++) {
        std::copy(imgstack[i].begin(), imgstack[i].end(), retval.it_slice(i));
    }
    return retval;
}

template <class T> std::vector<image2d<T>> slice_image(image3d<T> imgstack) {
    auto retval = std::vector<image2d<T>>(imgstack.nz(), image2d<T>(imgstack.nx(), imgstack.ny()));
    for (int i = 0; i < imgstack.nz(); i++) {
        std::copy(imgstack.it_slice(i), imgstack.it_slice(i + 1), retval[i].begin());
    }

    return retval;
}

template <class T>
image2d<T> crop_image(const image2d<T> &img, int xmin, int xmax, int ymin, int ymax) {
    int nx = xmax - xmin;
    int ny = ymax - ymin;

    assert(nx <= img.nx());
    assert(ny <= img.ny());
    assert(nx > 0);
    assert(ny > 0);

    image2d<T> cropped_img = image2d<T>(nx, ny);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            cropped_img(i, j) = img((i + xmin), (j + ymin));
        }
    }
    return cropped_img;
}

template <class T>
void crop_image_inplace(image2d<T> &img, int xmin, int xmax, int ymin, int ymax) {
    auto cropped_img = crop_image(img, xmin, xmax, ymin, ymax);
    std::swap(img, cropped_img);
}

template <class T> image2d<T> bin_image(image2d<T> img) {
    // TODO fix to extrapolate for odd size images
    int nx = img.nx();
    int ny = img.ny();

    assert(nx % 2 == 0);
    assert(ny % 2 == 0);

    int nx2 = nx / 2;
    int ny2 = ny / 2;

    auto binned = image2d<T>(nx2, ny2);
    for (int j = 0; j < ny2; j++) {
        for (int i = 0; i < nx2; i++) {
            binned(i, j) = img(2 * i + 1, 2 * j);
            binned(i, j) += img(2 * i + 1, 2 * j + 1);
            binned(i, j) += img(2 * i, 2 * j + 1);
            binned(i, j) += img(2 * i, 2 * j);
            binned(i, j) /= 4.0;
        }
    }

    return binned;
}

template <class T> void bin_image_inplace(image2d<T> &img) {
    auto binned = bin_image(img);
    std::swap(img, binned);
}
