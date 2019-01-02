#pragma once

#include <mpi.h>

typedef struct param {
    char stackfname[100];
    char anglefname[100];
    int haveangles;
    int cropx;
    int cropy;
    int xcent;
    int ycent;
    float tolsirt;
    int maxsirt;
    float lam;
    int maxit;
    int lcut;
    int rcut;
    int fudgefactor;
    char voutfname[100];
    int rmethod;
    int pmethod;
    float thresh;
    int align;
} arecparam;

void print_hint();
int parse_keyvalue(arecparam *param, const char *key, const char *value, int &havedata);
int parseinput(MPI_Comm comm, char *filename, arecparam *param);
int parseinput_old(MPI_Comm comm, char *filename, arecparam *param);
void setparams(arecparam inparam, int nx, int ny, double *radius, int *height, int *xcent,
               int *ycent, char *voutfname, int *lcut, int *rcut, int *fudgefactor, int *rmethod,
               int *pmethod);
