// Prealign2023.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>
#include <limits>
#include "TubeAlign.h"
#include "./NCXT-image/image2d.h"

const float eps = std::numeric_limits<float>::epsilon();
typedef std::vector<image2d<float>> image_stack;

float get_minimum_recorded(image2d<float>& img) {
    float threshold = 1e-6f;
    float value = 1e6f;

    for (auto val : img) {
        if (val < value && val > threshold) { value = val; }
    }
    return value;
}


void preProcessing(image_stack& img) {

    auto minus_logarithm = [](auto& x) { x = -log(x); };
    for (auto& im : img) {

        float minval = get_minimum_recorded(im);
        auto constain_positive = [minval](auto& x) {
            if (x < eps) x = minval;
        };

        im.apply(constain_positive);
        im.apply(minus_logarithm);
    }
}

void zeroToEdges(image_stack& img, TubeAlign t, double sigma) {
    t.updatecapillary_data();

    int nx = img[0].nx();
    int ny = img[0].ny();
    int nz = static_cast<int>(img.size());
    for (int zi = 0; zi < nz; zi++) {
        /* disregard outside data (overexposed parts)
        by smoothing it out towards zero */
        double divS = pow(10, 2);
        for (int yi = 0; yi < ny; yi++) {
            int x_min = static_cast<int>(round(t.getEdgeLeft(zi, yi) - 2 * sigma));
            int x_max = static_cast<int>(round(t.getEdgeRight(zi, yi) + 2 * sigma));

            for (int xi = 0; xi < x_min; xi++) {
                double dx = (xi - x_min);
                double weight = exp(-dx * dx / divS);
                img[zi].m_data[nx * yi + xi] *= static_cast<float>(weight);
            }
            for (int xi = x_max; xi < nx; xi++) {
                double dx = (xi - x_max);
                double weight = exp(-dx * dx / divS);
                img[zi].m_data[nx * yi + xi] *= static_cast<float>(weight);
            }
        }
    }
}

void postProcessing(image_stack& image, TubeAlign t, bool transmission, bool bin) {
    int nx = image[0].nx();
    int ny = image[0].ny();
    // zero absorption outside tube
    // zeroToEdges(image, t, 10.0);

    // choose largest volume from either translation bounds or edge bounds
    int crop_x = t.crop_limit_x(nx);



    //printf("Crop x: &d --  getMinEdgeSkip: %d getMaxAbsX: %d \n", crop_x,t.getMinEdgeSkip(), t.getMaxAbsX());


    int cropsize = (image[0].nx() - 2 * crop_x);
    // original arec required odd-sized images for imagecyl
    // this ensures that the binned image is odd in sizes
    if (cropsize % 4 == 0) crop_x++;

    // positive sY relates to movement towards 0, i.e., crop at end and vice versa.
    int crop_ya = max(0, t.getMaxNegY()) + 10;
    int crop_yb = max(0, t.getMaxPosY()) + 10;

    printf("Cropping images %d-%d %d-%d\n", crop_x, nx - crop_x, crop_ya, ny - crop_yb);
    auto exp_minus = [](auto& x) { x = exp(-1.0f * x); };
    for (auto& im : image) {
        crop_image_inplace(im, crop_x, nx - crop_x, crop_ya, ny - crop_yb);

        if (bin)
        {
            bin_image_inplace(im);
        }   
        if (transmission)
        {
            im.apply(exp_minus);
        }
    }
}



