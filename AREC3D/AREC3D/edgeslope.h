#pragma once

#include <mpi.h>

#include "arecImage.h"

int edgeslope(MPI_Comm comm, arecImage *images, int imgnum, float *angle, float thresh);
