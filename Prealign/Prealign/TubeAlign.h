#pragma once
#include "./NCXT-image/image2d.h"
#include "./NCXT-image/image3d.h"
#include "./NCXT-image/image_utils.h"
#include "./NCXT-image/mrc_file.h"
#include "TubeAlign_utils.h"
#include <algorithm>
#include <memory>
#include <vector>

using namespace std;

struct capillary_data {
    // fitting parameters
    double p1_left{}, p2_left{}, p1_err_left{};
    double p1_right{}, p2_right{}, p1_err_right{};
    // limits
    int min_left{}, max_right{};
    double COM{};

    capillary_data() {}
};

typedef std::vector<image2d<float>> image_stack;

class TubeAlign {
  private:
    const double _PI = 3.14159265359;
    // PARAMETERS FOR EDGE DETECTION
    const double edgeTH = 1.0;  // distance threshold (from the line) for RANSAC fitting
    const double cleanTH = 1.0; // distance threshold (from the line) for cleaning up the edge

    const double n_sigma = 3.0; // Sets the width of the kernel window in sigmas
    const double line_th = 0.5; // Percent of maximum to be considered as edge

    vector<float> _edge_kernel;
    vector<float> _profile_kernel;
    int _edge_halfwidth;
    int _profile_halfwidth;

    int PAD_TUBE_EDGE =
        10; // the amount of pixels to discard for the y-alignment (should be >~ tube thickness)
    int Y_PROFILE_PAD_VERTICAL = 50; // the amount of pixels to discard for the y-alignment

    // kernel parameters for profile
    const double p_n_sigm = 5.0;
    const double p_sigma = 3.0;

    // pixels to skip before searching for edge
    int edge_skip_left = 10;
    int edge_skip_right = 10;

  public:
    TubeAlign(image_stack &t_img, std::vector<double> &t_ang);
    ~TubeAlign();

  private:
    // containers for transformations
    vector<double> rot;
    vector<double> sX;
    vector<double> sY;
    // edge slopes
    vector<capillary_data> tube;
    // containers for edge locations
    vector<double> left_edge;
    vector<double> right_edge;

    image_stack &m_img;
    std::vector<double> &m_ang;

    bool debug_mode = false;

  public:
    void initialize_kernels(double sigma_canny, double sigma_profile);
    void set_padding(int pad_x, int pad_y);

    void savePreview(std::string file_name);

    double globalRotationFromCylinder();
    vector<int> getXTranslationsFromCylinder();

    double profile_gc_img(image2d<float> &yprofiles);
    double profile_cc_img(image2d<float> &yprofiles);

    image2d<float> getProfiles();
    vector<int> getYTranslationsFromProfile();
    vector<int> getXTranslationsFromProfile();

    int getMinEdgeSkip() { return min(edge_skip_left, edge_skip_right); }
    int getMaxAbsX() { return static_cast<int>(getMaxAbs(sX.data(), sX.size())); }
    int getMaxNegY() { return static_cast<int>(-getMinValue(sY.data(), sY.size())); }
    int getMaxPosY() { return static_cast<int>(getMaxValue(sY.data(), sY.size())); }
    int getMaxNegX() { return static_cast<int>(-getMinValue(sX.data(), sX.size())); }
    int getMaxPosX() { return static_cast<int>(getMaxValue(sX.data(), sX.size())); }

    void updateMaxSkips();

    void updatecapillary_data();
    double getEdgeLeft(int zi, int yi);
    double getEdgeRight(int zi, int yi);

    void debug() { debug_mode = true; }

    int crop_limit_x(int width);

  private:
    capillary_data getFittedSlopes(image_stack &t_img, int i);
    void findEdgesOfLine(image_stack &img, int zi, int yi, int &e1, int &e2);
};
