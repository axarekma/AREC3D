/* for MRC files only at the moment */
/* Does not check for endian!*/

#include "arecImageIO.h"
#include "arecImage.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int arecReadImageHeader(char *fname, int *nx, int *ny, int *nz) {
    MRCheader mrcheader;
    int status = 0, nbytes;

    FILE *fp;
    fp = fopen(fname, "rb");
    if (!fp) {
        fprintf(stderr, "failed to open %s\n", fname);
        status = -1;
        goto EXIT;
    }
    nbytes = fread(&mrcheader, sizeof(MRCheader), 1, fp);
    *nx = mrcheader.nx;
    *ny = mrcheader.ny;
    *nz = mrcheader.nz;
    fclose(fp);
EXIT:
    return status;
}

int arecReadImage(char *fname, arecImage *image) {
    int status = 0;
    int nx, ny, nz, mode, imagesize;
    int i, nbytes;
    short *sdata;
    MRCheader mrcheader;

    FILE *fp;
    fp = fopen(fname, "rb");
    if (!fp) {
        fprintf(stderr, "failed to open %s\n", fname);
        status = -1;
        goto EXIT;
    }
    nbytes = fread(&mrcheader, sizeof(MRCheader), 1, fp);
    nx = mrcheader.nx;
    ny = mrcheader.ny;
    nz = mrcheader.nz;
    image->nx = nx;
    image->ny = ny;
    image->nz = nz;

    mode = mrcheader.mode;

    imagesize = nx * ny * nz;
    image->data = (float *)malloc(imagesize * sizeof(float));
    if (mode == 1 || mode == 6) {
        sdata = (short *)malloc(imagesize * sizeof(short));
        if (fread(sdata, sizeof(short), imagesize, fp) != imagesize) {
            fprintf(stderr, "failed to read %s\n", fname);
            status = -1;
            free(sdata);
            goto EXIT;
        }
        for (i = 0; i < imagesize; i++)
            image->data[i] = (float)sdata[i];
        free(sdata);
    } else if (mode == 2) {
        if (fread(image->data, sizeof(float), imagesize, fp) != imagesize) {
            fprintf(stderr, "arecReadImage: failed to read %s\n", fname);
            status = -1;
        }
    } else {
        fprintf(stderr, "arecReadImage: invalid mrc mode: %d\n", mode);
        status = -1;
    }

EXIT:
    fclose(fp);
    return status;
}

int arecWriteImage(char *fname, arecImage image) {
    int nx, ny, nz;
    float *data;
    MRCheader mrcheader;
    double amin = 1.0e20, amax = -1.0e20, mean = 0.0;
    FILE *fp;
    int i, imagesize;
    int status = 0;

    nx = image.nx;
    ny = image.ny;
    nz = image.nz;
    data = image.data;
    imagesize = nx * ny * nz;

    mrcheader.nx = nx;
    mrcheader.ny = ny;
    mrcheader.nz = nz;
    mrcheader.mode = 2;
    mrcheader.nxstart = 0;
    mrcheader.nystart = 0;
    mrcheader.nzstart = 0;
    mrcheader.mx = nx;
    mrcheader.my = ny;
    mrcheader.mz = nz;

    mrcheader.xlen = 1; /* Cell dimensions (Angstroms). */
    mrcheader.ylen = 1; /* Cell dimensions (Angstroms). */
    mrcheader.zlen = 1; /* Cell dimensions (Angstroms). */

    mrcheader.alpha = 90.0; /* Cell angles (Degrees). */
    mrcheader.beta = 90.0;  /* Cell angles (Degrees). */
    mrcheader.gamma = 90.0; /* Cell angles (Degrees). */

    mrcheader.mapc = 1; /* Which axis corresponds to Columns.  */
    mrcheader.mapr = 2; /* Which axis corresponds to Rows.     */
    mrcheader.maps = 3;

    mean = 0.0;
    for (i = 0; i < imagesize; i++) {
        if (data[i] > amax) amax = data[i];
        if (data[i] < amin) amin = data[i];
        mean = mean + data[i];
    }
    mean = mean / (float)(nx * ny * nz);
    mrcheader.amin = amin;
    mrcheader.amax = amax;
    mrcheader.amean = mean;

    mrcheader.ispg = 0;   /* Space group number (0 for images). */
    mrcheader.nsymbt = 0; /* Number of chars used for storing symmetry */
                          /* operators.                                */
    for (i = 0; i < 25; i++)
        mrcheader.user[i] = ' ';
    mrcheader.user[24] = '\0';

    mrcheader.xorigin = 0;
    mrcheader.yorigin = 0;
    mrcheader.zorigin = 0;

    mrcheader.map[0] = 'M';
    mrcheader.map[1] = 'A';
    mrcheader.map[2] = 'P';
    mrcheader.map[3] = '\0';

    mrcheader.rms = 1.0;
    mrcheader.nlabels = 0;

    fp = fopen(fname, "wb");
    if (!fp) {
        fprintf(stderr, "failed to open %s\n", fname);
        status = -1;
        goto EXIT;
    }
    if (fwrite(&mrcheader, sizeof(MRCheader), 1, fp) != 1) {
        fprintf(stderr, "failed to write an MRC header\n");
        status = -1;
        goto EXIT;
    }

    if (fwrite(data, sizeof(float), imagesize, fp) != imagesize) {
        fprintf(stderr, "failed to write MRC data\n");
        status = -1;
        fclose(fp);
        goto EXIT;
    }
    fclose(fp);
EXIT:
    return status;
}
