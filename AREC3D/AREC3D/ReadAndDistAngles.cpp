#include "arecConstants.h"
#include "string.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int countrows(char *fname);
int ReadAndDistAngles(MPI_Comm gcomm, char *angfname, int nimgs, float *angles, int *iflip);

#define FNAMELENGTH 200

int countrows(char *fname) {
    char line[2000];
    int nangles;
    FILE *fp;

    fp = fopen(fname, "rb");
    if (!fp) {
        fprintf(stderr, "%s does not exist!\n", fname);
        return -1;
    }
    nangles = 0;
    while (fgets(line, 2000, fp) != NULL) {
        nangles++;
    }
    fclose(fp);
    return nangles;
}

int ReadAndDistAngles(MPI_Comm gcomm, char *angfname, int nimgs, float *angles, int *iflip) {
    /* read an angle file that contains a list of angles
       and distribute them evenly among different processors
    nimgs  --- total number of images in the original image data stack
    angles --- pointers to local arrays that contain distributed angles
    iflip  --- pointer to the index of the image that is the mirror image
                                                                                                                                       of the first image, i.e., the
    angle difference between the iflip-th image and the 1st image should be around 180 degrees.
    */

    int mypid, ierr = 0, nangles, ncpus, i;
    double t0;
    float angle0 = 0.0, mindf = 180.0;
    FILE *fp = NULL;

    MPI_Comm_rank(gcomm, &mypid);
    MPI_Comm_size(gcomm, &ncpus);
    *iflip = 0;
    int keep_searching = 1;

    if (mypid == 0) nangles = countrows(angfname);
    MPI_Bcast(&nangles, 1, MPI_INT, 0, gcomm);
    if (nangles != nimgs) {
        if (mypid == 0) {
            fprintf(stderr, "nangles and nimgs do not agree!\n");
            fprintf(stderr, "nangles = %d, nimgs = %d\n", nangles, nimgs);
        }
        return -1;
    }

    t0 = MPI_Wtime();
    if (mypid == 0) {
        fp = fopen(angfname, "rb");
        if (!fp) ierr = 1;
    }
    MPI_Bcast((void *)&ierr, 1, MPI_INT, 0, gcomm);
    if (ierr) {
        if (mypid == 0) fprintf(stderr, "failed to open %s\n", angfname);
        goto EXIT;
    } else {
        if (mypid == 0) {
            for (i = 0; i < nangles; i++) {
                if (fscanf(fp, "%f", &angles[i])) {}
                if (i == 0) {
                    if (angles[i] > 180.0) {
                        printf("Changing angle0 of %d from %4.2f", nangles, angles[i]);
                        angle0 = angles[i] - 360.0;
                        printf(" to %4.2f \n", angle0);
                    }
                    angle0 = angles[i];
                }
                if (keep_searching) {
                    if (fabs(angles[i] - angle0 - 180.0) < mindf) {
                        mindf = fabs(angles[i] - angle0 - 180.0);
                        (*iflip)++;
                    }
                }
                /* quit searching when 180 degree is passed */
                if (angles[i] - angle0 > 180.0) keep_searching = 0;
            } /*end for */
        }     /*end if*/
        MPI_Bcast(angles, nangles, MPI_FLOAT, 0, gcomm);
        MPI_Bcast(iflip, 1, MPI_INT, 0, gcomm);
    }
    if (mypid == 0) { printf("I/O time for reading angles = %11.3e\n", MPI_Wtime() - t0); }
    /* convert angles to radian */
    for (i = 0; i < nangles; i++)
        angles[i] = angles[i] / 180.0 * piFunc();
EXIT:
    return ierr;
}
