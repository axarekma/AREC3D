#include "align2dstack.h"
#include "arecConstants.h"
#include "arecImage.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define imagedata(i, j, k) imagedata[nx * ny * (k) + nx * (j) + (i)]
#define TOP 100
#define BOT ny - TOP

int edgeslope(MPI_Comm comm, arecImage *image, int imgnum, float *angle, float THRESH) {
    /* Determine the rotation angle from the slopes of the tube edges
       imgnum is the global image number to be processed */
    float *line;
    int tleft, tright, bleft, bright, i, ncpus, mypid, isum, ip;
    int nx, ny, nz, owner, imgnumloc, status;
    float s1, s2, a1, a2;
    int *count, *countloc;

    MPI_Comm_size(comm, &ncpus);
    MPI_Comm_rank(comm, &mypid);

    /* image data */
    float *imagedata;

    nx = image->nx;
    ny = image->ny;
    nz = image->nz;
    imagedata = image->data;
    status = 0;

    /* figure out which processor owns the imgnum-th image */
    countloc = (int *)calloc(ncpus, sizeof(int));
    count = (int *)calloc(ncpus, sizeof(int));
    countloc[mypid] = nz;

    /* collect image count from all processors */
    MPI_Allreduce(countloc, count, ncpus, MPI_INT, MPI_SUM, comm);
    owner = 0;
    isum = 0;
    for (ip = 0; ip < ncpus; ip++) {
        imgnumloc = imgnum - isum;
        isum = isum + count[ip];
        if (imgnum < isum) {
            if (mypid == ip) owner = 1;
            break;
        }
    }

    if (owner) {
        line = (float *)calloc(nx, sizeof(float));

        /* find left/right edge in the top line */
        for (i = 0; i < nx; i++) {
            if (imagedata(i, TOP, imgnumloc) > THRESH) {
                line[i] = THRESH;
            } else {
                line[i] = imagedata(i, TOP, imgnumloc);
            }
        }

        for (i = 0; i < nx - 3; i++) {
            if ((line[i] - line[i + 1]) * (line[i + 1] - line[i + 2]) < 0) {
                tleft = i;
                break;
            }
        }

        if (i == nx - 3) {
            fprintf(stderr, "failed to find the left edge at the top\n");
            status = -1;
            goto EXIT;
        }

        for (i = nx - 1; i >= 2; i--) {
            if ((line[i] - line[i - 1]) * (line[i - 1] - line[i - 2]) < 0) {
                tright = i;
                break;
            }
        }

        /* find left/right edge in the bottom line */
        for (i = 0; i < nx; i++)
            line[i] = 0.0;

        for (i = 0; i < nx; i++) {
            if (imagedata(i, BOT, imgnumloc) > THRESH) {
                line[i] = THRESH;
            } else {
                line[i] = imagedata(i, BOT, imgnumloc);
            }
        }

        for (i = 0; i < nx - 3; i++) {
            if ((line[i] - line[i + 1]) * (line[i + 1] - line[i + 2]) < 0) {
                bleft = i;
                break;
            }
        }

        if (i == nx - 3) {
            printf("failed to find the left edge at the bottom\n");
            status = -1;
            goto EXIT;
        }

        for (i = nx - 1; i >= 2; i--) {
            if ((line[i] - line[i - 1]) * (line[i - 1] - line[i - 2]) < 0) {
                bright = i;
                break;
            }
        }

#ifdef DEBUG
        printf("tleft, bleft, tright, bright = %d, %d, %d, %d\n", tleft, bleft, tright, bright);
#endif

        s1 = (float)(tleft - bleft) / (float)(ny - 2.0 * TOP);
        s2 = (float)(tright - bright) / (float)(ny - 2.0 * TOP);

        a1 = atan(s1);
        a2 = atan(s2);
#ifdef DEBUG
        printf("a1 = %f, a2 = %f\n", a1, a2);
#endif

        *angle = (a1 + a2) / 2.0 * 180.0 / PI;

        free(line);
    } /* endif owner */

EXIT:
    MPI_Bcast(&status, 1, MPI_INT, ip, comm);
    MPI_Bcast(angle, 1, MPI_FLOAT, ip, comm);
    free(countloc);
    free(count);
    return status;
}
#undef imagedata
