#pragma once
#include <mpi.h>

#include "arecImage.h"

int init_weights(MPI_Comm comm, arecImage projections, arecImage *weights);

int cyl_cgls_stat(MPI_Comm comm, arecImage images, float *angles, arecImage *xvol, float lam,
                  int maxit, float tol);
int cyl_cgls_KB_stat(MPI_Comm comm, arecImage images, float *angles, arecImage *xvol, float lam,
                     int maxit, float tol);
int cyl_cgls_SQ_stat(MPI_Comm comm, arecImage images, float *angles, arecImage *xvol, float lam,
                     int maxit, float tol);
