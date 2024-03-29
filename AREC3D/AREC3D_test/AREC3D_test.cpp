#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "arecImage.h"
#include "arecproject.h"
#include <iostream>

TEST_CASE("Cylvol, odd:", "[cylvol]") {
    arecImage cylvol;
    arecAllocateCylImage(&cylvol, 1, 2);
    int *cord = cylvol.cord;

    // X 0 X   --> 0
    // 1 2 3
    // X 4 X

    SECTION("Right amount of elements") {
        REQUIRE(cylvol.nrays == 5);
        REQUIRE(cylvol.nnz == 10);
    }

    SECTION("Correct coordinates") {
        REQUIRE(cord(0, 0) == 1);
        REQUIRE(cord(1, 0) == 0);

        REQUIRE(cord(0, 1) == 0);
        REQUIRE(cord(1, 1) == 1);

        REQUIRE(cord(0, 2) == 1);
        REQUIRE(cord(1, 2) == 1);

        REQUIRE(cord(0, 3) == 2);
        REQUIRE(cord(1, 3) == 1);

        REQUIRE(cord(0, 4) == 1);
        REQUIRE(cord(1, 4) == 2);
    }

    arecImageFree(&cylvol);
}

TEST_CASE("Cylvol, even:", "[cylvol]") {
    arecImage cylvol;
    arecAllocateCylImage(&cylvol, 0.5, 3);
    int *cord = cylvol.cord;

    // 0 1   --> 0
    // 2 3

    SECTION("Right amount of elements") {
        REQUIRE(cylvol.nrays == 4);
        REQUIRE(cylvol.nnz == 12);
    }

    SECTION("Correct coordinates") {
        REQUIRE(cord(0, 0) == 0);
        REQUIRE(cord(1, 0) == 0);

        REQUIRE(cord(0, 1) == 1);
        REQUIRE(cord(1, 1) == 0);

        REQUIRE(cord(0, 2) == 0);
        REQUIRE(cord(1, 2) == 1);

        REQUIRE(cord(0, 3) == 1);
        REQUIRE(cord(1, 3) == 1);
    }

    arecImageFree(&cylvol);
}

TEST_CASE("projection, data:", "[proj]") {
    arecImage projstack;
    arecAllocateCBImage(&projstack, 1, 2, 3);

    int nx = projstack.nx;
    int ny = projstack.ny;

    for (size_t i = 0; i < 1 * 2 * 3; i++) {
        projstack.data[i] = 1.0f * i;
    }
    float *cube = projstack.data;

    REQUIRE(cube(0, 0, 0) == Approx(0.0));
    REQUIRE(cube(0, 1, 0) == Approx(1.0));
    REQUIRE(cube(0, 0, 1) == Approx(2.0));
    REQUIRE(cube(0, 1, 1) == Approx(3.0));
    REQUIRE(cube(0, 0, 2) == Approx(4.0));
    REQUIRE(cube(0, 1, 2) == Approx(5.0));

    // s0 0 s1 2 s2 4
    //    1    3    5

    arecImageFree(&projstack);
}

TEST_CASE("projection, odd:", "[proj]") {
    arecImage cylvol;
    arecImage projstack;
    arecAllocateCylImage(&cylvol, 1, 2);
    arecAllocateCBImage(&projstack, 3, 2, 1);

    cylvol.data[4] = 1.0f;
    cylvol.data[5] = 2.0f;

    // cylvol
    // X 0 X   --> 0
    // 1 2 3
    // X 4 X
    int *cord = cylvol.cord;

    SECTION("Right amount of elements") {
        REQUIRE(cylvol.nrays == 5);
        REQUIRE(cylvol.nnz == 10);
    }

    // proj
    // 0 1 2
    // 3 4 5

    float angle = 0.0f;
    int status = arecProject2D_SQ(cylvol, &angle, 1, &projstack);

    int nx = projstack.nx;
    int ny = projstack.ny;
    int nz = projstack.nz;
    double sum_volume = 0.0;
    for (int i = 0; i < cylvol.nnz; i++) {
        sum_volume += cylvol.data[i];
    }
    double sum_projection = 0.0;
    for (int i = 0; i < nx * ny * nz; i++) {
        sum_projection += projstack.data[i];
    }

    SECTION("check sum_vol") { REQUIRE(sum_volume == Approx(3.0)); }
    SECTION("check sum_proj") { REQUIRE(sum_projection == Approx(3.0)); }

    SECTION("check pixels") {

        REQUIRE(projstack.data[1] == Approx(1.0));
        REQUIRE(projstack.data[4] == Approx(2.0));
    }

    arecImageFree(&cylvol);
    arecImageFree(&projstack);
}

TEST_CASE("backprojection, odd:", "[proj]") {
    arecImage cylvol;
    arecImage projstack;
    arecAllocateCylImage(&cylvol, 1, 2);
    arecAllocateCBImage(&projstack, 3, 2, 1);

    projstack.data[1] = 1.0;
    projstack.data[4] = 2.0;

    // proj
    // 0 1 2
    // 3 4 5

    float angle = 0.0f;
    int status = arecBackProject2D_SQ(projstack, &angle, 1, &cylvol);

    int nx = projstack.nx;
    int ny = projstack.ny;
    int nz = projstack.nz;

    double sum_volume = 0.0;
    for (int i = 0; i < cylvol.nnz; i++) {
        sum_volume += cylvol.data[i];
    }
    double sum_projection = 0.0;
    for (int i = 0; i < nx * ny * nz; i++) {
        sum_projection += projstack.data[i];
    }

    SECTION("check sum_vol") { REQUIRE(sum_volume == Approx(9.0)); }
    SECTION("check sum_proj") { REQUIRE(sum_projection == Approx(3.0)); }

    SECTION("check pixels") {

        REQUIRE(projstack.data[1] == Approx(1.0));
        REQUIRE(projstack.data[4] == Approx(2.0));
    }

    arecImageFree(&cylvol);
    arecImageFree(&projstack);
}

float coeff_A(arecImage &cyl, arecImage &proj, float *angles, int nangles, int a, int b) {
    reset(&cyl);
    reset(&proj);

    cyl.data[a] = 1.0f;
    int status = arecProject2D_SQ(cyl, angles, nangles, &proj);

    // print(&cyl);
    // print(&proj);

    return proj.data[b];
}
float coeff_AT(arecImage &proj, arecImage &cyl, float *angles, int nangles, int a, int b) {
    reset(&cyl);
    reset(&proj);

    proj.data[a] = 1.0f;
    int status = arecBackProject2D_SQ(proj, angles, nangles, &cyl);

    // print(&proj);
    // print(&cyl);

    return cyl.data[b];
}

TEST_CASE("A vs AT, odd:", "[proj]") {
    arecImage cylvol;
    arecImage projstack;

    int n_angles = 3;
    int ny = 2;
    double rad = 1.0;
    float *angles;
    angles = (float *)malloc(n_angles * sizeof(float));
    angles[0] = 1.475;
    angles[1] = 21.123;
    angles[2] = 2.123;
    arecAllocateCylImage(&cylvol, rad, ny);
    int nx = cylvol.nx;
    arecAllocateCBImage(&projstack, nx, ny, n_angles);

    printf("n_cyl: %d n_proj = %d\n", cylvol.nnz, projstack.nx * projstack.ny * projstack.nz);
    for (size_t a = 0; a < cylvol.nnz; a++) {
        for (size_t b = 0; b < projstack.nx * projstack.ny * projstack.nz; b++) {
            REQUIRE(coeff_A(cylvol, projstack, angles, n_angles, a, b) ==
                    coeff_AT(projstack, cylvol, angles, n_angles, b, a));
        }
    }

    arecImageFree(&cylvol);
    arecImageFree(&projstack);
    free(angles);
}

TEST_CASE("A vs AT, even:", "[proj]") {
    arecImage cylvol;
    arecImage projstack;
    int step = 3;
    int n_angles = 2;
    int ny = 2;
    double rad = 1.5;
    float *angles;
    angles = (float *)malloc(n_angles * sizeof(float));
    angles[0] = 0.698132;
    angles[1] = 1.22173;
    arecAllocateCylImage(&cylvol, rad, ny);
    int nx = cylvol.nx;
    arecAllocateCBImage(&projstack, nx, ny, n_angles);

    printf("n_cyl: %d n_proj = %d\n", cylvol.nnz, projstack.nx * projstack.ny * projstack.nz);
    for (size_t a = 0; a < cylvol.nnz; a += step) {
        for (size_t b = 0; b < projstack.nx * projstack.ny * projstack.nz; b += step) {
            REQUIRE(coeff_A(cylvol, projstack, angles, n_angles, a, b) ==
                    coeff_AT(projstack, cylvol, angles, n_angles, b, a));
        }
    }

    arecImageFree(&cylvol);
    arecImageFree(&projstack);
    free(angles);
}

float coeff_A_KB(arecImage &cyl, arecImage &proj, float *angles, int nangles, int a, int b) {
    reset(&cyl);
    reset(&proj);

    cyl.data[a] = 1.0f;
    int status = arecProject2D_KB(cyl, angles, nangles, &proj);

    // print(&cyl);
    // print(&proj);

    return proj.data[b];
}
float coeff_AT_KB(arecImage &proj, arecImage &cyl, float *angles, int nangles, int a, int b) {
    reset(&cyl);
    reset(&proj);

    proj.data[a] = 1.0f;
    int status = arecBackProject2D_KB(proj, angles, nangles, &cyl);

    // print(&proj);
    // print(&cyl);

    return cyl.data[b];
}

TEST_CASE("A vs AT, even KB:", "[kb]") {
    arecImage cylvol;
    arecImage projstack;
    int step = 3;
    int n_angles = 2;
    int ny = 2;
    double rad = 0.5;
    float *angles;
    angles = (float *)malloc(n_angles * sizeof(float));
    angles[0] = 0.698132;
    angles[1] = 1.22173;
    arecAllocateCylImage(&cylvol, rad, ny);
    int nx = cylvol.nx;
    arecAllocateCBImage(&projstack, nx, ny, n_angles);

    printf("n_cyl: %d n_proj = %d\n", cylvol.nnz, projstack.nx * projstack.ny * projstack.nz);
    for (size_t a = 0; a < cylvol.nnz; a += step) {
        for (size_t b = 0; b < projstack.nx * projstack.ny * projstack.nz; b += step) {
            REQUIRE(coeff_A_KB(cylvol, projstack, angles, n_angles, a, b) ==
                    coeff_AT_KB(projstack, cylvol, angles, n_angles, b, a));
        }
    }

    arecImageFree(&cylvol);
    arecImageFree(&projstack);
    free(angles);
}

TEST_CASE("ImageCyl2CB", "[cylvol]") {
    arecImage cylvol;
    arecAllocateCylImage(&cylvol, 1, 2);
    int *cord = cylvol.cord;

    int length = cylvol.nnz;
    for (size_t i = 0; i < length; i++) {
        cylvol.data[i] = 1.0 * i;
    }
    double sum_cyle = 0.0;
    for (int i = 0; i < cylvol.nnz; i++) {
        sum_cyle += cylvol.data[i];
    }

    arecImage xcbvol;
    arecAllocateCBImage(&xcbvol, cylvol.nx, cylvol.ny, cylvol.nz);

    ImageCyl2CB(cylvol, xcbvol, cylvol.nx, cylvol.ny, cylvol.nz);

    length = xcbvol.nx * xcbvol.ny * xcbvol.nz;
    double sum_cub = 0.0;
    for (int i = 0; i < length; i++) {
        sum_cub += xcbvol.data[i];
    }

    SECTION("check sum_vol") { REQUIRE(sum_cub == sum_cyle); }

    arecImage cylvol2;
    arecAllocateCylImage(&cylvol2, 1, 2);
    ImageCB2Cyl(xcbvol, cylvol2, 2, 2, 2, 2);

    length = cylvol.nnz;
    for (size_t i = 0; i < length; i++) {
        REQUIRE(cylvol.data[i] == cylvol2.data[i]);
    }

    arecImageFree(&cylvol);
}
