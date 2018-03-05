#include "TubeAlign.h"
#include "TubeAlign_utils.h"

TubeAlign::TubeAlign(image_stack &t_img, std::vector<double> &t_ang) : m_img(t_img), m_ang(t_ang) {
    sX = vector<double>(t_img.size(), 0.0);
    sY = vector<double>(t_img.size(), 0.0);
    rot = vector<double>(t_img.size(), 0.0);
    tube = vector<capillary_data>(t_img.size());
}

TubeAlign::~TubeAlign() {}

void TubeAlign::savePreview(std::string file_name) {
    updatecapillary_data();

    image_stack temp = image_stack(m_img);

    for (size_t iz = 0; iz < temp.size(); iz++) {
        float amax =
            static_cast<float>(*std::max_element(temp[iz].m_data.begin(), temp[iz].m_data.end()));

        for (int yi = 0; yi < temp[iz].ny(); yi++) {
            int x_min = static_cast<int>(std::round(tube[iz].p1_left * yi + tube[iz].p2_left));
            int x_max = static_cast<int>(std::round(tube[iz].p1_right * yi + tube[iz].p2_right));

            temp[iz](x_min, yi) = amax;
            temp[iz](x_max, yi) = amax;

            temp[iz](x_min - 1, yi) = 0;
            temp[iz](x_max - 1, yi) = 0;
            temp[iz](x_min + 1, yi) = 0;
            temp[iz](x_max + 1, yi) = 0;
        }
    }

    writeImage_stream(combine_stack(temp), file_name);
}

double TubeAlign::globalRotationFromCylinder() {

    printf(" capillary_data temp = getFittedSlopes(m_img, i); \n");
    updatecapillary_data();

    // setup arrays for fitting
    vector<double> slope1(m_img.size());
    vector<double> slope2(m_img.size());
    vector<double> mslope(m_img.size());
    vector<double> slope1_err(m_img.size());
    vector<double> slope2_err(m_img.size());
    vector<double> mslope_err(m_img.size());

    for (size_t i = 0; i < m_img.size(); ++i) {
        slope1[i] = atan(tube[i].p1_left);
        slope2[i] = atan(tube[i].p1_right);
        mslope[i] = (slope1[i] + slope2[i]) / 2.0;
        slope1_err[i] =
            tube[i].p1_err_left; // actually dydx is sec2(x) but tan is close enough to 1...
        slope2_err[i] = tube[i].p1_err_right;
        mslope_err[i] = 0.5 * sqrt(slope1_err[i] * slope1_err[i] + slope2_err[i] * slope2_err[i]);
    }

    vector<double> p1 = {0.0, 0.0, 0.0};
    vector<double> p2 = {0.0, 0.0, 0.0};
    vector<double> pm = {0.0, 0.0, 0.0};
    double val1, val2, valm;

    fitRotation(m_ang, slope1, slope1_err, p1, val1, false);
    fitRotation(m_ang, slope2, slope2_err, p2, val2, false);
    fitRotation(m_ang, mslope, mslope_err, pm, valm, false);
    /*
    Restart with best initial guess.
    In testing, sometimes only one of these found a good fit from ig (0,0,0)
    */
    if (val1 < valm) valm = val1;
    if (val2 < valm) valm = val2;

    printf("Restarting fit with initial guess from best fit of slopes.\n");

    fitRotation(m_ang, mslope, mslope_err, pm, valm, true);

    return -pm[2];
}

vector<int> TubeAlign::getXTranslationsFromCylinder() {
    vector<int> retval(m_img.size(), 0);
    double x_center = 0.5 * (m_img[0].nx() - 1); // more general for all size stakcs?

    updatecapillary_data();

    // set up translations for x-direction
    for (size_t i = 0; i < m_img.size(); i++) {
        double com_x = tube[i].COM;
        double minleft = tube[i].min_left;
        double maxright = tube[i].max_right;
        if (minleft > maxright)
            printf("Slopes in wrong order at [%zu] %4.2f  %4.2f--%4.2f]\n ", i, com_x, minleft,
                   maxright);

        sX[i] = round(com_x - x_center);
        retval[i] = static_cast<int>(round(com_x - x_center));
    }

    return retval;
}

// global cross corelation of the profiles
double TubeAlign::profile_gc_img(image2d<float> &yprofiles) {
    int prof_x = yprofiles.nx();
    int prof_y = yprofiles.ny();

    double ymin, ymax;

    auto abs_comp = [](double a, double b) { return (std::fabs(a) < std::fabs(b)); };
    int pad = static_cast<int>(std::fabs(*std::max_element(sY.begin(), sY.end(), abs_comp)) +
                               Y_PROFILE_PAD_VERTICAL);

    printf("Doing y-aling GC with PAD=%d\n", pad);
    int cc_length = prof_x - 2 * pad;

    std::vector<float> sumprofile = std::vector<float>(prof_x);
    std::vector<double> sY_temp = std::vector<double>(prof_y);
    // printf("profile_gc prof_x=%d prof_y=%d \n",prof_x,prof_y);

    // get sumprofile
    for (int j = 0; j < prof_y; j++) {
        for (int i = 0; i < prof_x; i++)
            sumprofile[i] += yprofiles[prof_x * j + i] / prof_y;
    }

    // correlate to sumpprofile
    for (int j = 0; j < prof_y; j++) {
        std::vector<float> p_cccoefs = std::vector<float>(cc_length);
        ccorr1d(cc_length, &yprofiles[prof_x * j + pad], &sumprofile[pad], p_cccoefs.data());
        sY_temp[j] = findMaxCC_fit(p_cccoefs.data(), cc_length);
        double shift = std::fabs(sY_temp[j]);
        if (shift > 10.0) {

            sY_temp[j] *= pad;
            sY_temp[j] /= shift * shift;
            printf("index [%d] shift is suspiciously large. Damping %f --> %f \n", j,
                   sY_temp[j] / fabs(sY_temp[j]) * shift, sY_temp[j]);
        }
    }

    ymin = getMinValue(sY_temp.data(), prof_y);
    ymax = getMaxValue(sY_temp.data(), prof_y);
    double y_mean = getMeanValue(sY_temp.data(), prof_y);

    for (int j = 0; j < prof_y; j++) {
        // sY_temp[j] -= 0.5 * (ymax + ymin);
        sY_temp[j] -= y_mean;
        circShiftX(&yprofiles[prof_x * j], prof_x, static_cast<int>(std::round(sY_temp[j])));
        sY[j] -= sY_temp[j];
    }

    return fabs(ymax - ymin);
}

// succenssive cross corelation of the profiles
double TubeAlign::profile_cc_img(image2d<float> &yprofiles) {
    int prof_x = yprofiles.nx();
    int prof_y = yprofiles.ny();

    double ymin, ymax;
    std::vector<double> sY_temp = std::vector<double>(prof_y);

    auto abs_comp = [](double a, double b) { return (std::fabs(a) < std::fabs(b)); };
    int pad = static_cast<int>(std::fabs(*std::max_element(sY.begin(), sY.end(), abs_comp)) +
                               Y_PROFILE_PAD_VERTICAL);

    printf("Doing y-aling GC with PAD=%d\n", pad);
    int cc_length = prof_x - 2 * pad;

    sY_temp[0] = 0.0;
    for (int j = 1; j < prof_y; j++) {
        std::vector<float> p_cccoefs = std::vector<float>(cc_length);
        ccorr1d(cc_length, &yprofiles[prof_x * j + pad], &yprofiles[prof_x * (j - 1) + pad],
                p_cccoefs.data());
        sY_temp[j] = findMaxCC_fit(p_cccoefs.data(), cc_length);
    }
    // integrate
    for (int j = 1; j < prof_y; j++)
        sY_temp[j] += sY_temp[j - 1];

    ymin = getMinValue(sY_temp.data(), prof_y);
    ymax = getMaxValue(sY_temp.data(), prof_y);
    double y_mean = getMeanValue(sY_temp.data(), prof_y);

    for (int j = 0; j < prof_y; j++) {
        // sY_temp[j] -= 0.5 * (ymax + ymin);
        sY_temp[j] -= y_mean;

        circShiftX(&yprofiles[prof_x * j], prof_x, +static_cast<int>(std::round(sY_temp[j])));
        sY[j] -= sY_temp[j];
        // if (j>90) printf("j: %d s=%4.2f  sg=%4.2f \n",j,sY_temp[j],sY[j]);
    }
    // printf("Y shifts between %4.2f and %4.2f
    // \n",getMinValue(sY_temp,prof_y),getMaxValue(sY_temp,prof_y));

    return fabs(ymax - ymin);
}
//}

image2d<float> TubeAlign::getProfiles() {
    int ny = static_cast<int>(m_img[0].ny());
    int nz = static_cast<int>(m_img.size());

    int prof_x{m_img[0].ny()}; // assuming all images are the same size
    int prof_y = static_cast<int>(m_img.size());

    image2d<float> yprofiles = image2d<float>(prof_x, prof_y);

    updatecapillary_data();

    for (int iz = 0; iz < nz; iz++) {
        for (int yi = 0; yi < ny; yi++) {
            // edge1 is x=a*y+c
            // edge2 is x=b*y+d
            int x_min = static_cast<int>(std::round(tube[iz].p1_left * yi + tube[iz].p2_left) +
                                         PAD_TUBE_EDGE);
            int x_max = static_cast<int>(std::round(tube[iz].p1_right * yi + tube[iz].p2_right) -
                                         PAD_TUBE_EDGE);
            double line_mean = getLineMean(m_img, iz, yi, x_min, x_max);

            yprofiles[ny * iz + yi] = static_cast<float>(line_mean);
        }

        /*For some cases the density profile doesn't work
        usually because of distinct background profiles.
        Better results were obtained by using the 1st derivative.*/
        int halfkernel = static_cast<int>(std::floor(p_n_sigm * p_sigma));
        vector<float> kernel = getGaussianDerivateKernel(p_n_sigm, p_sigma);
        convolve_valid_inplace(&yprofiles[ny * iz], kernel.data(), 0.0, ny, halfkernel);
    }
    return yprofiles;
}

vector<int> TubeAlign::getYTranslationsFromProfile() {
    printf("======================================\n");
    printf("Aligning y-shifts with density profile\n");
    printf("======================================\n");
    // Store profiles in array
    int prof_y = static_cast<int>(m_img.size());

    vector<double> profile_sX(m_img.size(), 0.0);
    vector<double> profile_sY(m_img.size(), 0.0);

    vector<image2d<float>> stack_profile_cc_img;
    vector<image2d<float>> stack_profile_gc_img;

    image2d<float> yprofiles = getProfiles();
    saveProfile(yprofiles, "profile_RAW_new.png");

    double val;
    stack_profile_cc_img.push_back(yprofiles);
    while ((val = profile_cc_img(yprofiles)) > 1.0) {
        stack_profile_cc_img.push_back(yprofiles);
        printf("Yalign_cc max shift is %4.2f pixels\n", val);
    }

    printf("Yalign_cc max shift is %4.2f pixels\n", val);
    stack_profile_gc_img.push_back(yprofiles);
    while ((val = profile_gc_img(yprofiles)) > 1.0) {
        stack_profile_gc_img.push_back(yprofiles);
        printf("Yalign max shift is %4.2f pixels\n", val);
    }
    printf("Yalign max shift is %4.2f pixels\n", val);

    saveProfile(yprofiles, "profile_ALIGNED_new.png");

    writeImage_stream(combine_stack(stack_profile_cc_img), "stack_profile_cc_img.mrc");
    writeImage_stream(combine_stack(stack_profile_gc_img), "stack_profile_gc_img.mrc");

    vector<int> retval(m_img.size());
    for (size_t i = 0; i < retval.size(); i++) {
        retval[i] = static_cast<int>(std::round(sY[i]));
    }

    return retval;
}

vector<int> TubeAlign::getXTranslationsFromProfile() {
    vector<int> retval(m_img.size(), 0);
    // Preset sX to mean slope later to be multiplied by the sY;
    for (size_t i = 0; i < retval.size(); i++) {
        retval[i] = static_cast<int>(round(0.5 * (tube[i].p1_left + tube[i].p1_right) * sY[i]));
    }

    return retval;
}

void TubeAlign::updatecapillary_data() {
#pragma omp parallel for // differs from serial because random order in RANSAC, checked
    for (int i = 0; i < m_img.size(); ++i) {
        tube[i] = getFittedSlopes(m_img, i);
    }

    int edge_skip_left_new = m_img[0].nx();
    int edge_skip_right_new = m_img[0].nx();

    for (int i = 0; i < m_img.size(); ++i) {
        edge_skip_left_new =
            min(edge_skip_left_new, tube[i].min_left - static_cast<int>(sigma * 2 * n_sigma));
        edge_skip_right_new =
            min(edge_skip_right_new, static_cast<int>(m_img[0].nx()) - 1 - tube[i].max_right -
                                         static_cast<int>(sigma * 2 * n_sigma));
    }
    edge_skip_left = max(edge_skip_left, edge_skip_left_new);
    edge_skip_right = max(edge_skip_right, edge_skip_right_new);

    printf("new edge skips %d %d\n", edge_skip_left, edge_skip_right);
}

double TubeAlign::getEdgeLeft(int zi, int yi) { return tube[zi].p1_left * yi + tube[zi].p2_left; }

double TubeAlign::getEdgeRight(int zi, int yi) {
    return tube[zi].p1_right * yi + tube[zi].p2_right;
}

capillary_data TubeAlign::getFittedSlopes(image_stack &t_img, int i) {
    std::vector<int> edge1 = std::vector<int>(t_img[0].ny());
    std::vector<int> edge2 = std::vector<int>(t_img[0].ny());

    double p1_left, p1_right, p2_left, p2_right;
    double p1_left_err, p1_right_err;
    capillary_data res;

    // find edges of tube
    for (int yi = 0; yi < t_img[0].ny(); yi++) {
        int e1, e2;
        findEdgesOfLine(t_img, i, yi, e1, e2);
        edge1[yi] = e1;
        edge2[yi] = e2;
    }

    // Clean edge detection with RANSAC
    ransac(edge1.data(), t_img[0].ny(), 1000, edgeTH, p1_left, p2_left);
    ransac(edge2.data(), t_img[0].ny(), 1000, edgeTH, p1_right, p2_right);
    cleanEdge(edge1.data(), t_img[0].ny(), cleanTH, p1_left, p2_left);
    cleanEdge(edge2.data(), t_img[0].ny(), cleanTH, p1_right, p2_right);

    // Linear fit with confidence bounds on slope
    linFitEdgeWerr(edge1.data(), t_img[0].ny(), p1_left, p2_left, p1_left_err);
    linFitEdgeWerr(edge2.data(), t_img[0].ny(), p1_right, p2_right, p1_right_err);

    // gather result
    res.p1_left = p1_left;
    res.p2_left = p2_left;
    res.p1_err_left = p1_left_err;
    res.p1_right = p1_right;
    res.p2_right = p2_right;
    res.p1_err_right = p1_right_err;

    int left_top = static_cast<int>(std::round(p1_left * 0 + p2_left));
    int right_top = static_cast<int>(std::round(p1_right * 0 + p2_right));

    int left_bottom = static_cast<int>(std::round(p1_left * (t_img[0].ny() - 1) + p2_left));
    int right_bottom = static_cast<int>(std::round(p1_right * (t_img[0].ny() - 1) + p2_right));

    res.min_left = min(left_top, left_bottom);
    res.max_right = max(right_top, right_bottom);
    res.COM = 0.25 * (left_top + left_bottom) + 0.25 * (right_top + right_bottom);

    return res;
}

void TubeAlign::findEdgesOfLine(image_stack &img, int zi, int yi, int &e1, int &e2) {
    int halfkernel = static_cast<int>(std::floor(n_sigma * sigma));

    vector<float> vec_buffer = convolve_valid(&img[zi].m_data[yi * img[zi].nx()],
                                              _edge_kernel.data(), 0.0f, img[zi].nx(), halfkernel);
    float const *buffer = vec_buffer.data();

    // find minmax
    double min_val = buffer[halfkernel];
    double max_val = buffer[halfkernel];
    int min_ind = halfkernel;
    int max_ind = halfkernel;
    double l_lim = (1.0 / 3.0) * img[zi].nx();
    double r_lim = (2.0 / 3.0) * img[zi].nx();

    // todo use min_el
    // auto test = std::min_element( vec_buffer.begin() + edge_skip_left, vec_buffer.end() -
    // edge_skip_right) ;

    for (int j = edge_skip_left; j < img[zi].nx() - edge_skip_right; j++) {
        // if (yi==500) printf("%d,%4.2f \n",j,buffer[j]);
        if (buffer[j] < min_val && j > l_lim) {
            min_val = buffer[j];
            min_ind = j;
        }
        if (buffer[j] > max_val && j < r_lim) {
            max_val = buffer[j];
            max_ind = j;
        }
    }
    // if lin
    // e1=min_ind;
    // e2=max_ind;
    // if log
    e2 = min_ind;
    e1 = max_ind;

    // cleanup clearly wrong edges
    // double l_lim = (1.0 / 3.0)*img->nx;
    // double r_lim = (2.0 / 3.0)*img->nx;
    if (e2 < l_lim) e2 = -1;
    if (e1 > r_lim) e1 = -1;

    // delete[] X;
    // 0delete[] Y;
    // delete[] buffer;
}
