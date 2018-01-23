#pragma once
#include <mpi.h>

#include "arecImage.h"

int cyl_cgls_KB(MPI_Comm comm, arecImage images, float *angles, arecImage *xvol, float lam,
                int maxit, float tol);
