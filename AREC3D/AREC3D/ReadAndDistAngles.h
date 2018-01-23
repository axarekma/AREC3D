#pragma once

#include <mpi.h>

int ReadAndDistAngles(MPI_Comm gcomm, char *angfname, int nimgstot, float *angles, int *iflip);
