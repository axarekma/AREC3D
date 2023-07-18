// Prealign_2017.cpp : Defines the entry point for the console application.
//

#include "fftw3.h"
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "./NCXT-image/image2d.h"
#include "./NCXT-image/image3d.h"
#include "./NCXT-image/image_utils.h"
#include "./NCXT-image/mrc_file.h"

#include "Prealign.h"
#include "TubeAlign.h"
#include "TubeAlign_utils.h"
#include "mpfit.h"

#include "cxxopts.hpp"


using namespace std;

cxxopts::options cxxopts_setup(char *const argv[]) {
    try {
        cxxopts::options options(argv[0], "One line description of MyProgram");

        options.positional_help("projections.mrc angles.rawtlt output.mrc").show_positional_help();

        options.add_options() //
            ("positional",
             "Input files in the order\n"
             "  projections.mrc: input stack\n"
             "  angles.rawtlt: angles [deg]\n"
             "  output.mrc: output file",
             cxxopts::value<std::vector<std::string>>()) //
            ("h,help", "Print help");

        options.add_options("Alignment") //
            ("t,transmission", "Projection images in [t]ransmission",
             cxxopts::value<bool>()->default_value("true"))(
                "canny", "sigma for Canny edge detection",
                cxxopts::value<double>()->default_value("5.0")) //
            ("sigma", "sigma for Common Line profile",
             cxxopts::value<double>()->default_value("5.0")) //
            ("c,capillary", "Thickness discarded for Common Line profile",
             cxxopts::value<int>()->default_value("30")) //
            ("pad", "Vertical padding for Common Line profile",
             cxxopts::value<int>()->default_value("50")) //
            ("b,bin", "[B]in output image", cxxopts::value<bool>()->default_value("true"));
        options.parse_positional({"positional"});

        return options;

    } catch (const cxxopts::option_error &e) {
        std::cout << "error incxxopts_setup: " << e.what() << '\n';
        std::cout << "Press Enter to Continue";
        std::cin.ignore();
        exit(1);
    }
}

int main(int argc, char *argv[]) {

    bool debug_mode = false;
    std::vector<char *> argv_debug;
    string base =
        "c:/Users/axela/Documents/2023/SiriusXT/capillary/20230620_capillary_new_shroud_axel/";
    string infile = (base + "20230620_capillary_new_shroud_tlt_flat_arec.mrc");
    string inangle = (base + "20230620_capillary_new_shroud.rawtlt");
    string outfile = (base + "20230620_capillary_new_shroud_tlt_flat_pre.mrc");
    std::vector<std::string> arguments = {infile, inangle, outfile};

    argv_debug.push_back(argv[0]);
    for (const auto &arg : arguments)
        argv_debug.push_back((char *)arg.data());
    argv_debug.push_back(nullptr);

    if (debug_mode) {
        argc = 4;
        argv = argv_debug.data();
    }
    cxxopts::options options = cxxopts_setup(argv);

    try {
        cxxopts::parse_result parameters = options.parse(argc, argv);

        if (parameters.count("help")) {
            std::cout << options.help({"Alignment"}) << '\n';
            std::cout << "Press Enter to Continue";
            std::cin.ignore();
            exit(0);
        }

        if (parameters.count("positional") != 3) {
            std::cout << "Invalid positional argumens!\n";
            std::cout << "Got " << parameters.count("positional") << ", wanted 3!\n";

            std::cout << options.help() << '\n';
            std::cout << "Press Enter to Continue";
            std::cin.ignore();
            exit(0);
        }

        std::vector<std::string> str_load = parameters["positional"].as<std::vector<std::string>>();
        image_stack projections = slice_image(arecReadImage_stream(str_load[0]));
        std::vector<double> angles = loadAngles(str_load[1]);

        if (projections.size() == 0) {
            cout << "Error in reading mrc file";
            std::cout << str_load[0];
            return 0;
        }
        if (angles.size() < 1) {
            std::cout << "Failed to load angles.";
            return 0;
        }
        if (angles.size() != projections.size()) {
            std::cout << "Size of stack (";
            std::cout << projections.size();
            std::cout << ") and angles (";
            std::cout << angles.size();
            std::cout << ") do not match.";
            return 0;
        }

        vector<TransformMatrix2D> M_final = vector<TransformMatrix2D>(projections.size());
        TubeAlign t = TubeAlign(projections, angles);

        // Set parameters
        t.initialize_kernels(parameters["canny"].as<double>(), parameters["sigma"].as<double>());
        t.set_padding(parameters["capillary"].as<int>(), parameters["pad"].as<int>());

        if (parameters["transmission"].as<bool>()) {
            std::cout << "Preprocessing Transmission Images\n";
            preProcessing(projections);
        }
        t.savePreview("initial_canny_edge.mrc");
        double global_rot = t.globalRotationFromCylinder(); // in radians
        TransformMatrix2D M_rot = rotation_matrix_rad(global_rot);
#pragma omp parallel for
        for (int i = 0; i < projections.size(); i++) {
            chain_transormations_inplace(M_final[i], M_rot);
            transform_clip_inplace(projections[i], M_rot);
        }

        vector<int> sX = t.getXTranslationsFromCylinder();
#pragma omp parallel for
        for (int i = 0; i < projections.size(); i++) {
            TransformMatrix2D M_trans = translation_matrix(-sX[i], 0);
            transform_clip_inplace(projections[i], M_trans);
            chain_transormations_inplace(M_final[i], M_trans);
        }
        t.updateMaxSkips();

        vector<int> sY = t.getYTranslationsFromProfile();
        vector<int> sX2 = t.getXTranslationsFromProfile();
#pragma omp parallel for
        for (int i = 0; i < projections.size(); i++) {
            TransformMatrix2D M_trans = translation_matrix(-sX2[i], -sY[i]);
            transform_clip_inplace(projections[i], M_trans);
            M_final[i] = chain_transormations(M_final[i], M_trans);
        }

        postProcessing(projections, t, parameters["transmission"].as<bool>(),
                       parameters["bin"].as<bool>());

        printf("Writing log, image and input file for AREC\n");
        writePrealignment("prealign_log.txt", global_rot, sX, sX2, sY);
        writeInput(str_load[2].c_str(), str_load[1].c_str());
        writeAngles(str_load[1].c_str(), angles);
        writeImage_stream(combine_stack(projections), str_load[2].c_str());

    } catch (const cxxopts::option_error &e) {
        std::cout << "error parsing options: " << e.what() << '\n';
        std::cout << "Press Enter to Continue";
        std::cin.ignore();
        exit(1);
    }

    return 0;
}
