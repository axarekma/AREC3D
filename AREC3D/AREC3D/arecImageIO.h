#pragma once

#include "arecImage.h"

enum {
    MRC_NUM_LABELS = 10,
    MRC_LABEL_SIZE = 80,
    NUM_4BYTES_PRE_MAP = 52,
    NUM_4BYTES_AFTER_MAP = 3
};

typedef struct header {
    int nx; /* number of columns */
    int ny; /* number of rows */
    int nz; /* number of sections */

    int mode; /* See modes above. */

    int nxstart; /* No. of first column in map, default 0. */
    int nystart; /* No. of first row in map, default 0. */
    int nzstart; /* No. of first section in map,default 0. */

    int mx; /* Number of intervals along X. */
    int my; /* Number of intervals along Y. */
    int mz; /* Number of intervals along Z. */

    /* Cell: treat a whole 2D image as a cell */
    float xlen; /* Cell dimensions (Angstroms). */
    float ylen; /* Cell dimensions (Angstroms). */
    float zlen; /* Cell dimensions (Angstroms). */

    float alpha; /* Cell angles (Degrees). */
    float beta;  /* Cell angles (Degrees). */
    float gamma; /* Cell angles (Degrees). */

    /* axis X => 1, Y => 2, Z => 3 */
    int mapc;    /* Which axis corresponds to Columns.  */
    int mapr;    /* Which axis corresponds to Rows.     */
    int maps;    /* Which axis corresponds to Sections. */
    float amin;  /* Minimum density value. */
    float amax;  /* Maximum density value. */
    float amean; /* Mean density value.    */

    int ispg; /* Space group number (0 for images). */

    int nsymbt; /* Number of chars used for storing symmetry */
                /* operators.                                */

    int user[25];

    float xorigin; /* X origin. */
    float yorigin; /* Y origin. */
    float zorigin; /* Y origin. */

    char map[4];      /* constant string "MAP "  */
    int machinestamp; /* machine stamp in CCP4 convention: */
                      /* big endian=0x11110000 little endian=0x4444000 */

    float rms; /* rms deviation of map from mean density */

    int nlabels; /* Number of labels being used. */
    char labels[MRC_NUM_LABELS][MRC_LABEL_SIZE];
} MRCheader;

int arecReadImageHeader(char *fname, int *nx, int *ny, int *nz);
int arecReadImage(char *fname, arecImage *image);
int arecWriteImage(char *fname, arecImage image);
