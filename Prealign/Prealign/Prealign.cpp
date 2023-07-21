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

template<typename T>
void print_arg(cxxopts::parse_result result, std::string tag, std::string desc)
{
	int const printwidth = 30;
	if (result.count(tag)) {

		std::cout << std::setw(printwidth)
			<< desc << " = " << result[tag].as<T>() << '\n';
	}
	else
	{
		std::cout << std::setw(printwidth)
			<< desc << " = " << result[tag].as<T>() << " (default)\n";
	}
}

void printargs(cxxopts::parse_result result) {
	int const printwidth = 30;
	std::cout << "Input files \n";
	if (result.count("input")) {
		std::cout << std::setw(printwidth) << "Input = " << result["input"].as<std::string>() << '\n';
	}
	if (result.count("angles")) {
		std::cout << std::setw(printwidth) << "Angles = " << result["angles"].as<std::string>() << '\n';
	}
	if (result.count("output")) {
		std::cout << std::setw(printwidth) << "Output = " << result["output"].as<std::string>() << '\n';
	}

	std::cout << "Parameters \n";
	print_arg<int>(result, "transmission", "Transmission File");
	print_arg<double>(result, "canny", "Canny sigma");
	print_arg<double>(result, "sigma", "Profile Sigma");
	print_arg<int>(result, "capillary", "Capillary thickness");
	print_arg<int>(result, "pad", "Y padding");
	print_arg<int>(result, "bin", "Binning");

	return;
}


cxxopts::options cxxopts_setup(char* const argv[]) {
	try {
		cxxopts::options options(argv[0], "One line description of MyProgram");

		options.positional_help("projections.mrc angles.rawtlt output.mrc").show_positional_help();

		options.add_options()    //
			("input", ".mrc projection stack", cxxopts::value<std::string>())
			("angles", ".rawtlt angles (degrees)", cxxopts::value<std::string>())
			("output", ".mrc output file", cxxopts::value<std::string>())
			("h,help", "Print help");


		options.add_options("Alignment") //
			("t,transmission", "Projection images in [t]ransmission",
				cxxopts::value<int>()->default_value("1"))
			("canny", "sigma for Canny edge detection",
				cxxopts::value<double>()->default_value("5.0"))   //
			("sigma", "sigma for Common Line profile",
				cxxopts::value<double>()->default_value("5.0"))   //
			("c,capillary", "Thickness discarded for Common Line profile",
				cxxopts::value<int>()->default_value("30"))  //      
			("pad", "Vertical padding for Common Line profile",
				cxxopts::value<int>()->default_value("50"))  //      
			("b,bin", "[B]in output image",
				cxxopts::value<int>()->default_value("1"))
			;
		options.parse_positional({ "input", "angles", "output" });

		return options;

	}
	catch (const cxxopts::option_error& e) {
		std::cout << "error incxxopts_setup: " << e.what() << '\n';
		std::cout << "Press Enter to Continue";
		std::cin.ignore();
		exit(1);
	}
}

int main(int argc, char *argv[]) {

    cxxopts::options options = cxxopts_setup(argv);
    try {
        cxxopts::parse_result parameters = options.parse(argc, argv);

        if (parameters.count("help")) {
            std::cout << options.help({"Alignment"}) << '\n';
            std::cout << "Press Enter to Continue";
            std::cin.ignore();
            exit(0);
        }

		if (parameters.count("input") + parameters.count("angles") + parameters.count("output") != 3) {
			std::cout << "Invalid positional argumens!\n";

			std::cout << "Got following arguments:\n";
			printargs(parameters);

			std::cout << options.help() << '\n';
			std::cout << "Press Enter to Continue";
			std::cin.ignore();
			exit(0);
		}

		std::cout << "Running Prealignment with arguments:\n\n";
		printargs(parameters);
		std::cout << "\n";


		std::string input_file = parameters["input"].as<std::string>();
		std::string angle_file = parameters["angles"].as<std::string>();
		std::string output_file = parameters["output"].as<std::string>();
		image_stack projections = slice_image(arecReadImage_stream(input_file));
		std::vector<double> angles = loadAngles(angle_file);

		if (projections.size() == 0) {
			cout << "Error in reading mrc file";
			std::cout << input_file;
			return 0;
		}
		if (angles.size() < 1) {
			std::cout << "Failed to load angles.";
			std::cout << angle_file;
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

        if (parameters["transmission"].as<int>()) {
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

        postProcessing(projections, t, parameters["transmission"].as<int>(),
                       parameters["bin"].as<int>());

		printf("Writing log, image and input file for AREC\n");
		writePrealignment("prealign_log.txt", global_rot, sX, sX2, sY);
		writeInput(output_file.c_str(), angle_file.c_str());
		writeAngles(angle_file.c_str(), angles);
		writeImage_stream(combine_stack(projections), output_file.c_str());

    } catch (const cxxopts::option_error &e) {
        std::cout << "error parsing options: " << e.what() << '\n';
        std::cout << "Press Enter to Continue";
        std::cin.ignore();
        exit(1);
    }

    return 0;
}
