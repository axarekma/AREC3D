#pragma once
#include <mpi.h>

#include "arecImage.h"

int cyl_sirt(MPI_Comm comm, arecImage images, float *angles, arecImage *xvol, float lam, int maxit,
             float tol);

int cyl_sirt_SQ(MPI_Comm comm, arecImage images, float *angles, arecImage *xvol, float lam,
                int maxit, float tol);
