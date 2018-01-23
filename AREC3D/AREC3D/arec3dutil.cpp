#include "arec3dutil.h"
#include "mpi.h"
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

/*---------------------------------------------------------*/
void print_hint() {
    printf("Not enough arguments to the command...\n");
    printf("Usage: runsirt <input file name>\n");
    printf("The input file contains the following information:\n");
    printf("data: the name of the MRC file that contains experimental images\n");
    printf("angles: the name of the ASCII file that contains angles associated with each image in "
           "the image data file\n");
    printf("cropx: the size of the cropped image in the x (row) direction\n");
    printf("cropy: the size of the cropped image in the y (column) direction\n");
    printf("lam: the damping parameter used in SIRT reconstruction\n");
    printf("tolsirt: the covergence tolerance used in SIRT reconstruction\n");
    printf("maxsirt: the maximum number of SIRT iterations allowed\n");
    printf("maxiter: the maximum number of alignment cycles allowed\n");
    printf("lcut: the number of pixels to be cut off from the left for alignment\n");
    printf("rcut: the number of pixels to be cut off from the right for alignment\n");
    printf("fudgefactor: the total number of pixels to be cut off ");
    printf("from both sides to create a cropped image for 2nd alignment");
    printf("rmethod: the method used to perform 3D reconstruction.");
    printf(" sirt --- 1, cgls --- 2\n");
    printf("pmethod: the method used to perform projections.");
    printf(" old --- 1, KB --- 2\n");
}

int parse_keyvalue(arecparam *param, const char *key, const char *value, int &havedata) {
    if (strncmp(key, "data", 4) == 0) {
        strcpy_s(param->stackfname, value);
        havedata = 1;
    } else if (strncmp(key, "angle", 5) == 0) {
        strcpy_s(param->anglefname, value);
        param->haveangles = 1;
    } else if (strncmp(key, "cropx", 5) == 0) {
        param->cropx = atoi(value);
    } else if (strncmp(key, "cropy", 5) == 0) {
        param->cropy = atoi(value);
    } else if (strncmp(key, "xcent", 5) == 0) {
        param->xcent = atoi(value);
    } else if (strncmp(key, "ycent", 5) == 0) {
        param->ycent = atoi(value);
    } else if (strncmp(key, "tolsirt", 7) == 0) {
        param->tolsirt = atof(value);
    } else if (strncmp(key, "maxsirt", 7) == 0) {
        param->maxsirt = atoi(value);
    } else if (strncmp(key, "lam", 3) == 0) {
        param->lam = atof(value);
    } else if (strncmp(key, "maxit", 5) == 0) {
        param->maxit = atoi(value);
    } else if (strncmp(key, "output", 6) == 0) {
        strcpy_s(param->voutfname, value);
    } else if (strncmp(key, "lcut", 4) == 0) {
        param->lcut = atoi(value);
    } else if (strncmp(key, "rcut", 4) == 0) {
        param->rcut = atoi(value);
    } else if (strncmp(key, "fudge", 5) == 0) {
        param->fudgefactor = atoi(value);
    } else if (strncmp(key, "rmethod", 6) == 0) {
        param->rmethod = atoi(value);
    } else if (strncmp(key, "pmethod", 6) == 0) {
        param->pmethod = atoi(value);
    } else if (strncmp(key, "thresh", 6) == 0) {
        param->thresh = atof(value);
    } else {
        fprintf(stderr, "invalid key in the input: %s\n", key);
        return -1;
    }
    return 0;
}

int parseinput(MPI_Comm comm, char *filename, arecparam *param) {
    int mypid;
    int status = 0;
    int havedata = 0;

    MPI_Comm_rank(comm, &mypid);

    /* initialize parameters */
    param->stackfname[1] = '\0';
    param->anglefname[1] = '\0';
    param->haveangles = 0;
    param->cropx = -1;
    param->cropy = -1;
    param->xcent = -1;
    param->ycent = -1;
    param->tolsirt = 1.0e-3;
    param->maxsirt = 10;
    param->lam = 5e-6;
    param->maxit = 1;
    param->lcut = -1;
    param->rcut = -1;
    param->fudgefactor = -1;
    param->rmethod = 2;
    param->pmethod = 1;
    param->thresh = 0.5;

    if (mypid == 0) {
        std::ifstream infile(filename);
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            // std::cout << line << '\n';

            int pos = line.find_first_of(' ');
            int pos2 = pos;
            while (line[pos2] == ' ')
                pos2++;
            std::string value = line.substr(pos2), key = line.substr(0, pos);
            status = parse_keyvalue(param, key.c_str(), value.c_str(), havedata);

            if (status != 0) { return status; }
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    if (status != 0) return status;

    MPI_Bcast(&havedata, 1, MPI_INT, 0, comm);
    if (!havedata) {
        fprintf(stderr, "The input file should at least contain the filename");
        fprintf(stderr, "that contains the experimental data!");
        status = -1;
        return status;
    }

    /* now broad cast param to other processors */
    MPI_Bcast(param->stackfname, 100, MPI_CHAR, 0, comm);
    MPI_Bcast(&param->haveangles, 1, MPI_INT, 0, comm);
    MPI_Bcast(param->anglefname, 100, MPI_CHAR, 0, comm);
    MPI_Bcast(&param->cropx, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->cropy, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->xcent, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->ycent, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->tolsirt, 1, MPI_FLOAT, 0, comm);
    MPI_Bcast(&param->maxsirt, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->lam, 1, MPI_FLOAT, 0, comm);
    MPI_Bcast(&param->maxit, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->lcut, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->rcut, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->fudgefactor, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->rmethod, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->pmethod, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->thresh, 1, MPI_FLOAT, 0, comm);

    return status;
}

int parseinput_old(MPI_Comm comm, char *filename, arecparam *param) {
    FILE *fp = NULL;
    int mypid;
    int status = 0;
    char key[100], value[100];
    int havedata = 0;

    MPI_Comm_rank(comm, &mypid);

    /* initialize parameters */
    param->stackfname[1] = '\0';
    param->anglefname[1] = '\0';
    param->haveangles = 0;
    param->cropx = -1;
    param->cropy = -1;
    param->xcent = -1;
    param->ycent = -1;
    param->tolsirt = 1.0e-3;
    param->maxsirt = 10;
    param->lam = 5e-6;
    param->maxit = 1;
    param->lcut = -1;
    param->rcut = -1;
    param->fudgefactor = -1;
    param->rmethod = 2;
    param->pmethod = 1;
    param->thresh = 0.5;

    if (mypid == 0) {
        errno_t err = fopen_s(&fp, filename, "rb");
        if (!fp) {
            fprintf(stderr, "failed to open %s\n", filename);
            status = -1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    if (status != 0) return status;

    if (mypid == 0) {
        while (fscanf_s(fp, "%s %s", key, value) != EOF) {
            if (strncmp(key, "data", 4) == 0) {
                strcpy_s(param->stackfname, value);
                havedata = 1;
            } else if (strncmp(key, "angle", 5) == 0) {
                strcpy_s(param->anglefname, value);
                param->haveangles = 1;
            } else if (strncmp(key, "cropx", 5) == 0) {
                param->cropx = atoi(value);
            } else if (strncmp(key, "cropy", 5) == 0) {
                param->cropy = atoi(value);
            } else if (strncmp(key, "xcent", 5) == 0) {
                param->xcent = atoi(value);
            } else if (strncmp(key, "ycent", 5) == 0) {
                param->ycent = atoi(value);
            } else if (strncmp(key, "tolsirt", 7) == 0) {
                param->tolsirt = atof(value);
            } else if (strncmp(key, "maxsirt", 7) == 0) {
                param->maxsirt = atoi(value);
            } else if (strncmp(key, "lam", 3) == 0) {
                param->lam = atof(value);
            } else if (strncmp(key, "maxit", 5) == 0) {
                param->maxit = atoi(value);
            } else if (strncmp(key, "output", 6) == 0) {
                strcpy_s(param->voutfname, value);
            } else if (strncmp(key, "lcut", 4) == 0) {
                param->lcut = atoi(value);
            } else if (strncmp(key, "rcut", 4) == 0) {
                param->rcut = atoi(value);
            } else if (strncmp(key, "fudge", 5) == 0) {
                param->fudgefactor = atoi(value);
            } else if (strncmp(key, "rmethod", 6) == 0) {
                param->rmethod = atoi(value);
            } else if (strncmp(key, "pmethod", 6) == 0) {
                param->pmethod = atoi(value);
            } else if (strncmp(key, "thresh", 6) == 0) {
                param->thresh = atof(value);
            } else {
                fprintf(stderr, "invalid key in the input: %s\n", key);
                status = -1;
                break;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    if (status != 0) return status;

    MPI_Bcast(&havedata, 1, MPI_INT, 0, comm);
    if (!havedata) {
        fprintf(stderr, "The input file should at least contain the filename");
        fprintf(stderr, "that contains the experimental data!");
        status = -1;
        return status;
    }

    /* now broad cast param to other processors */
    MPI_Bcast(param->stackfname, 100, MPI_CHAR, 0, comm);
    MPI_Bcast(&param->haveangles, 1, MPI_INT, 0, comm);
    MPI_Bcast(param->anglefname, 100, MPI_CHAR, 0, comm);
    MPI_Bcast(&param->cropx, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->cropy, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->xcent, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->ycent, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->tolsirt, 1, MPI_FLOAT, 0, comm);
    MPI_Bcast(&param->maxsirt, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->lam, 1, MPI_FLOAT, 0, comm);
    MPI_Bcast(&param->maxit, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->lcut, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->rcut, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->fudgefactor, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->rmethod, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->pmethod, 1, MPI_INT, 0, comm);
    MPI_Bcast(&param->thresh, 1, MPI_FLOAT, 0, comm);

    return status;
}

/* -----------------------------------------------*/
void setparams(arecparam inparam, int nx, int ny, int *radius, int *height, int *xcent, int *ycent,
               char *voutfname, int *lcut, int *rcut, int *fudgefactor, int *rmethod,
               int *pmethod) {
    /* set some parameters based on images size and input */

    if (inparam.cropx > 0) {
        *radius = (inparam.cropx - 1) / 2;
    } else {
        *radius = (nx - 1) / 2;
    }
    if (inparam.cropy > 0) {
        *height = inparam.cropy;
    } else {
        *height = ny;
    }
    if (inparam.xcent > 0) {
        *xcent = inparam.xcent;
    } else {
        *xcent = (nx - 1) / 2;
    }
    if (inparam.ycent > 0) {
        *ycent = inparam.ycent;
    } else {
        *ycent = (ny - 1) / 2;
    }
    if (inparam.voutfname != NULL) {
        strcpy_s(voutfname, 200, inparam.voutfname); // FIXME!
    } else {
        /* no output filename provided */
        strcpy_s(voutfname, 200, "myvolsirt.mrc"); // FIXME!
    }
    *lcut = inparam.lcut;
    *rcut = inparam.rcut;
    *fudgefactor = inparam.fudgefactor;
    *rmethod = inparam.rmethod;
    *pmethod = inparam.pmethod;
}
