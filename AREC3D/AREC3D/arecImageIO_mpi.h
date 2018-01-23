#pragma once

void arecReadImageDistbyZ(MPI_Comm comm, const char *fname, arecImage *image);
void arecReadImageDistbyY(MPI_Comm comm, const char *fname, arecImage *image);
void arecWriteImageMergeY(MPI_Comm comm, const char *fname, arecImage image);
void arecWriteImageMergeZ(MPI_Comm comm, const char *fname, arecImage image);
