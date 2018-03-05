// Prealign_2017.cpp : Defines the entry point for the console application.
//

#include "fftw3.h"
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "./NCXT-image/image2d.h"
#include "./NCXT-image/image3d.h"
#include "./NCXT-image/image_utils.h"
#include "./NCXT-image/mrc_file.h"

#include "TubeAlign.h"
#include "TubeAlign_utils.h"
#include "mpfit.h"

using namespace std;
const float eps = std::numeric_limits<float>::epsilon();
// imageMRC image_original;

typedef std::vector<image2d<float>> image_stack;

float get_minimum_recorded(image2d<float> &img) {
    float threshold = 1e-6;
    float value = 1e6;

    for (auto val : img) {
        if (val < value && val > threshold) { value = val; }
    }
    return value;
}

void preProcessing(image_stack &img) {

    auto remove_bad = [](auto &x) {
        if (std::isnan(x) || x > 1.0) x = 1.0;
    };
    auto minus_logarithm = [](auto &x) { x = -log(x); };

    // binImage(&image);
    for (auto &im : img) {
        float minval = get_minimum_recorded(im);

        auto constain_positive = [minval](auto &x) {
            if (x < eps) x = minval;
        };

        im.apply(remove_bad);
        im.apply(constain_positive);
        im.apply(minus_logarithm);
    }
}

void zeroToEdges(image_stack &img, TubeAlign t, double sigma) {
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

void postProcessing(image_stack &image, TubeAlign t) {
    int nx = image[0].nx();
    int ny = image[0].ny();
    // zero absorption oputside tube
    zeroToEdges(image, t, 10.0);

    // choose largest volume from either translation bounds or edge bounds
    int crop_x = max(0, min(t.getMinEdgeSkip(), t.getMaxAbsX()));

    int cropsize = (image[0].nx() - 2 * crop_x);
    // original arec required odd-szed images for imagecyl
    // this ensures that the binned image is odd in sizes
    if (cropsize % 4 == 0) crop_x++;

    // positive sY relates to movement towards 0, i.e., crop at end and vice versa.
    int crop_ya = max(0, t.getMaxPosY()) + 10;
    int crop_yb = max(0, t.getMaxNegY()) + 10;

    printf("Cropping images %d-%d %d-%d\n", crop_x, nx - crop_x, crop_ya, ny - crop_yb);
    auto exp_minus = [](auto &x) { x = exp(-1.0f * x); };
    for (auto &im : image) {
        crop_image_inplace(im, crop_x, nx - crop_x, crop_ya, ny - crop_yb);
        bin_image_inplace(im);
        im.apply(exp_minus);
    }
}

int main(int argc, char *argv[]) {
    bool debug_mode = false;

    if (argc != 4) // Check the value of argc. If not enough parameters have been passed, inform
                   // user and exit.
    {
        cout << "Found " << argc - 1
             << " argument(s). Need <infile> <inangle> <outfile>\n"; // Inform the user of how to
                                                                     // use the program
                                                                     // cin.get();
        return 0;
    }

    image_stack projections = slice_image(arecReadImage_stream(argv[1]));

    if (projections.size() == 0) {
        cout << "Error in reading file";
        // std::cin.get();
        return 1;
    }

    std::vector<double> angles = loadAngles(argv[2]);
    if (angles.size() < 1) {
        std::cout << "Failed to load angles. Press [Enter] to continue . . .";
        // std::cin.get();
        return 0;
    }

    if (angles.size() != projections.size()) {
        std::cout << "Stack and angles do not match. Press [Enter] to continue . . .";
        // std::cin.get();
        return 0;
    }

    vector<TransformMatrix2D> M_final = vector<TransformMatrix2D>(projections.size());
    TubeAlign t = TubeAlign(projections, angles);

    if (debug_mode) { t.debug(); }

    preProcessing(projections);
    if (debug_mode) { writeImage_stream(combine_stack(projections), "debug_preprocess.mrc"); }

    double global_rot = t.globalRotationFromCylinder(); // in radians

    TransformMatrix2D M_rot = rotation_matrix_rad(global_rot);

#pragma omp parallel for
    for (int i = 0; i < projections.size(); i++) {
        chain_transormations_inplace(M_final[i], M_rot);
        transform_clip_inplace(projections[i], M_rot);
    }

    if (debug_mode) { t.savePreview("debug_global_rot.mrc"); }

    vector<int> sX = t.getXTranslationsFromCylinder();
#pragma omp parallel for
    for (int i = 0; i < projections.size(); i++) {
        // printf("Transformm [%d], %d\n" , i, sX[i]);
        TransformMatrix2D M_trans = translation_matrix(-sX[i], 0);
        transform_clip_inplace(projections[i], M_trans);
        chain_transormations_inplace(M_final[i], M_trans);
    }

    if (debug_mode) { t.savePreview("debug_translation_tube.mrc"); }

    vector<int> sY = t.getYTranslationsFromProfile();
    vector<int> sX2 = t.getXTranslationsFromProfile();

#pragma omp parallel for
    for (int i = 0; i < projections.size(); i++) {
        TransformMatrix2D M_trans = translation_matrix(-sX2[i], -sY[i]);
        transform_clip_inplace(projections[i], M_trans);
        M_final[i] = chain_transormations(M_final[i], M_trans);
    }

    if (debug_mode) { t.savePreview("debug_translation_profile.mrc"); }

    // ofstream myfile;
    // myfile.open(manual_folder + "out\\translations_final.txt");
    // for (auto M : M_final) {
    //    myfile << M;
    //}
    // myfile.close();

    postProcessing(projections, t);

    // for (auto &im : projections) {
    // bin_image_inplace(im);
    //}

    printf("Writing log, image and input file for AREC\n");
    writePrealignment("prealign_log.txt", global_rot, sX, sX2, sY);
    writeInput_quick(argv[3], argv[2]);
    writeInput(argv[3], argv[2]);
    writeAngles(argv[2], angles);
    writeImage_stream(combine_stack(projections), argv[3]);
    /**/

    // printf("Done, press any key to exit. \n");
    // std::cin.get();
    return 0;
}
