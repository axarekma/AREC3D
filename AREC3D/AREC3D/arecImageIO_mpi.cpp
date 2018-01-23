#include "arecImage.h"
#include "arecImageIO.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

void arecReadImageDistbyZ(MPI_Comm comm, const char *fname, arecImage *image) {
    /*
     * Read a stack of images from the root and distribute the slices
     * as evenly as possible among all processors within the communication
     * group associated with the communicator comm.
     *
     * endian not checked in this version
     */
    int mypid, nprocs, tag;
    MPI_Comm_rank(comm, &mypid);
    MPI_Comm_size(comm, &nprocs);

    MRCheader mrcheader;
    int mode;
    short *sdata = nullptr;
    float *buffer;

    FILE *fp = nullptr;
    int nx, ny, nz, i, nzloc, nrem, nimgs2read, imagesize, ip, ierr = 0;
    MPI_Status mpistatus;

    if (mypid == 0) {
        errno_t err = fopen_s(&fp, fname, "rb");
        if (!fp) {
            fprintf(stderr, "failed to open %s\n", fname);
            MPI_Abort(comm, ierr);
        }
        if (fread(&mrcheader, sizeof(MRCheader), 1, fp) != 1) {
            fprintf(stderr, "failed to open %s\n", fname);
            MPI_Abort(comm, ierr);
        }
        nx = mrcheader.nx;
        ny = mrcheader.ny;
        nz = mrcheader.nz;
        mode = mrcheader.mode;
    }
    MPI_Bcast(&nx, 1, MPI_INT, 0, comm);
    MPI_Bcast(&ny, 1, MPI_INT, 0, comm);
    MPI_Bcast(&nz, 1, MPI_INT, 0, comm);
    nzloc = nz / nprocs;
    nrem = nz % nprocs;
    if (mypid < nrem) nzloc++;

    imagesize = nx * ny * nzloc;
    image->data = (float *)malloc(imagesize * sizeof(float));
    if (!image->data) {
        fprintf(stderr, "failed to allocate image data\n");
        MPI_Abort(comm, ierr);
    }

    if (mypid == 0) {
        buffer = (float *)malloc(nx * ny * nzloc * sizeof(float));
        if (!buffer) {
            fprintf(stderr, "failed to allocate buffer for reading\n");
            MPI_Abort(comm, ierr);
        }
        if (mode == 1 || mode == 6) {
            sdata = (short *)malloc(nx * ny * nzloc * sizeof(short));
            if (!sdata) {
                fprintf(stderr, "failed to allocate sdata for reading\n");
                MPI_Abort(comm, ierr);
            }
        }

        for (ip = 0; ip < nprocs; ip++) {
            nimgs2read = nz / nprocs;
            if (ip < nrem) nimgs2read++;

            /* read images into a buffer */
            imagesize = nx * ny * nimgs2read;
            if (mode == 1 || mode == 6) {
                if (fread(sdata, sizeof(short), imagesize, fp) != imagesize) {
                    fprintf(stderr, "failed to read short data for reading\n");
                    MPI_Abort(comm, ierr);
                }
                for (i = 0; i < imagesize; i++)
                    buffer[i] = (float)sdata[i];
            } else if (mode == 2) {
                if (fread(buffer, sizeof(float), imagesize, fp) != imagesize) {
                    fprintf(stderr, "failed to read mrc data\n");
                    MPI_Abort(comm, ierr);
                }
            } else {
                fprintf(stderr, "image dist: invalid mrc mode\n");
                MPI_Abort(comm, ierr);
            }

            /* send to other processors or copy to image->data */
            if (ip == 0) {
                /* copy from buffer to data */
                for (i = 0; i < imagesize; i++)
                    image->data[i] = buffer[i];
                image->nx = nx;
                image->ny = ny;
                image->nz = nzloc;
                image->cord = NULL;
                image->is_cyl = 0;
            } else {
                tag = 0;
                MPI_Send(buffer, imagesize, MPI_FLOAT, ip, tag, comm);
            }
        } // end for ip

        free(buffer);
        if (mode == 1 || mode == 6) free(sdata);
        fclose(fp);

    } else {
        tag = 0;
        MPI_Recv(image->data, imagesize, MPI_FLOAT, 0, tag, comm, &mpistatus);
        image->nx = nx;
        image->ny = ny;
        image->nz = nzloc;
        image->cord = NULL;
        image->is_cyl = 0;
    }
}

void arecReadImageDistbyY(MPI_Comm comm, const char *fname, arecImage *image) {
    /*
     * Read a stack of images from the root and distribute each image
     * along the Y direction as evenly as possible among all processors
     * within the communication group associated with the communicator comm.
     *
     * endian not checked in this version
     */

    int mypid, nprocs;
    MPI_Comm_rank(comm, &mypid);
    MPI_Comm_size(comm, &nprocs);

    MRCheader mrcheader;
    int mode;
    short *sdata = nullptr;

    FILE *fp = nullptr;
    int nx, ny, nz, ierr = 0, ix, iz, iproc;
    int nrem, nyloc, imagesize, num_images;
    int *nylocdim, *nyoffsets;
    float *imgbuf = nullptr;
    MPI_Status mpistatus;

    if (mypid == 0) {
        errno_t err = fopen_s(&fp, fname, "rb");
        if (!fp) {
            fprintf(stderr, "failed to open %s\n", fname);
            MPI_Abort(comm, ierr);
        }
        if (fread(&mrcheader, sizeof(MRCheader), 1, fp) != 1) {
            fprintf(stderr, "failed to open %s\n", fname);
            MPI_Abort(comm, ierr);
        }
        nx = mrcheader.nx;
        ny = mrcheader.ny;
        nz = mrcheader.nz;
        mode = mrcheader.mode;
        num_images = nz;
    }
    MPI_Bcast(&nx, 1, MPI_INT, 0, comm);
    MPI_Bcast(&ny, 1, MPI_INT, 0, comm);
    MPI_Bcast(&nz, 1, MPI_INT, 0, comm);
    MPI_Bcast(&mode, 1, MPI_INT, 0, comm);
    MPI_Bcast(&num_images, 1, MPI_INT, 0, comm);

    nrem = ny % nprocs;
    nylocdim = (int *)malloc(nprocs * sizeof(int));
    nyoffsets = (int *)malloc(nprocs * sizeof(int));
    for (iproc = 0; iproc < nprocs; iproc++) {
        nylocdim[iproc] = ny / nprocs;
        if (iproc < nrem) nylocdim[iproc]++;
    }
    nyoffsets[0] = 0;
    for (iproc = 1; iproc < nprocs; iproc++) {
        nyoffsets[iproc] = nyoffsets[iproc - 1] + nylocdim[iproc - 1];
    }
    nyloc = nylocdim[mypid];

    image->data = (float *)malloc(nx * nyloc * nz * sizeof(float));
    if (!image->data) {
        fprintf(stderr, "failed to allocate image data\n");
        MPI_Abort(comm, ierr);
    }

    imagesize = nx * ny;

    /* allocate buffer for reading */
    if (mypid == 0) {
        imgbuf = (float *)malloc(imagesize * sizeof(float));
        if (!imgbuf) {
            fprintf(stderr, "failed to allocate buffer for reading\n");
            MPI_Abort(comm, ierr);
        }
        if (mode == 1 || mode == 6) {
            sdata = (short *)malloc(imagesize * sizeof(short));
            if (!sdata) {
                fprintf(stderr, "failed to allocate short data for reading\n");
                MPI_Abort(comm, ierr);
            }
        }
    }

    /* read one slice at a time and distribute in y direction  */
    for (iz = 0; iz < nz; ++iz) {
        if (mypid == 0) {
            if (mode == 1 || mode == 6) {
                if (fread(sdata, sizeof(short), imagesize, fp) != imagesize) {
                    fprintf(stderr, "failed to read mrc data\n");
                    MPI_Abort(comm, ierr);
                }
                for (ix = 0; ix < imagesize; ix++)
                    imgbuf[ix] = (float)sdata[ix];
            } else if (mode == 2) {
                if (fread(imgbuf, sizeof(float), imagesize, fp) != imagesize) {
                    fprintf(stderr, "failed to read mrc data\n");
                    MPI_Abort(comm, ierr);
                }
            } else {
                fprintf(stderr, "image dist: invalid mrc mode\n");
                MPI_Abort(comm, ierr);
            }

            /* extract the local piece from the buffered image
               send the local piece to other processors */

            for (iproc = 1; iproc < nprocs; iproc++)
                MPI_Send(&imgbuf[nx * nyoffsets[iproc]], nx * nylocdim[iproc], MPI_FLOAT, iproc, 0,
                         comm);

            /* keep the first chunk for myself */
            for (ix = 0; ix < nx * nyloc; ix++)
                image->data[nx * nyloc * iz + ix] = imgbuf[ix];
        } else {
            MPI_Recv(&image->data[nx * nyloc * iz], nx * nyloc, MPI_FLOAT, 0, 0, comm, &mpistatus);

        } /* end if (mypid == 0) */
    }     /* end for iz */

    image->nx = nx;
    image->ny = nyloc;
    image->nz = nz;
    image->cord = NULL;
    image->is_cyl = 0;

    if (mypid == 0) fclose(fp);

    free(nylocdim);
    free(nyoffsets);
    if (mypid == 0) {
        free(imgbuf);
        if (mode == 1 || mode == 6) free(sdata);
    }
}

void arecWriteImageMergeY(MPI_Comm comm, const char *fname, arecImage image) {
    /*
     * collect image segments from all processors and the write out the image stack
     * to a single file. This write merges in the y direction.
     */

    MRCheader mrcheader;
    int ierr = 0;
    float amin = 1.0e20, amax = -1.0e20, mean = 0.0;
    FILE *fp = nullptr;
    float *imgbuf = nullptr;
    int datasize = 0, nx, ny, nz, nyloc, yoffset, i, ix, iz, iproc;
    int mypid, nproc;
    MPI_Status mpistatus;

    MPI_Comm_rank(comm, &mypid);
    MPI_Comm_size(comm, &nproc);

    nyloc = image.ny;
    ny = 0;
    MPI_Allreduce(&nyloc, &ny, 1, MPI_INT, MPI_SUM, comm);
    nx = image.nx;
    nz = image.nz;

    /* write header */
    mrcheader.nx = nx;
    mrcheader.ny = ny;
    mrcheader.nz = nz;
    mrcheader.mode = 2;
    mrcheader.nxstart = 0;
    mrcheader.nystart = 0;
    mrcheader.nzstart = 0;
    mrcheader.mx = nx;
    mrcheader.my = ny;
    mrcheader.mz = nz;

    mrcheader.xlen = 1; /* Cell dimensions (Angstroms). */
    mrcheader.ylen = 1; /* Cell dimensions (Angstroms). */
    mrcheader.zlen = 1; /* Cell dimensions (Angstroms). */

    mrcheader.alpha = 90.0; /* Cell angles (Degrees). */
    mrcheader.beta = 90.0;  /* Cell angles (Degrees). */
    mrcheader.gamma = 90.0; /* Cell angles (Degrees). */

    mrcheader.mapc = 1; /* Which axis corresponds to Columns.  */
    mrcheader.mapr = 2; /* Which axis corresponds to Rows.     */
    mrcheader.maps = 3;

    mean = 0.0;
    for (i = 0; i < nx * nyloc * nz; i++) {
        if (image.data[i] > amax) amax = image.data[i];
        if (image.data[i] < amin) amin = image.data[i];
        mean = mean + image.data[i];
    }

    MPI_Allreduce(&mean, &(mrcheader.amean), 1, MPI_FLOAT, MPI_SUM, comm);
    mrcheader.amean = mrcheader.amean / (float)(nx * ny * nz);
    MPI_Allreduce(&amax, &(mrcheader.amax), 1, MPI_FLOAT, MPI_MAX, comm);
    MPI_Allreduce(&amin, &(mrcheader.amin), 1, MPI_FLOAT, MPI_MIN, comm);

    mrcheader.ispg = 0;   /* Space group number (0 for images). */
    mrcheader.nsymbt = 0; /* Number of chars used for storing symmetry */
                          /* operators.                                */
    for (i = 0; i < 25; i++)
        mrcheader.user[i] = ' ';
    mrcheader.user[24] = '\0';

    mrcheader.xorigin = 0;
    mrcheader.yorigin = 0;
    mrcheader.zorigin = 0;

    mrcheader.map[0] = 'M';
    mrcheader.map[1] = 'A';
    mrcheader.map[2] = 'P';
    mrcheader.map[3] = '\0';

    mrcheader.rms = 1.0;
    mrcheader.nlabels = 0;

    /* write the header first */
    if (mypid == 0) {
        errno_t err = fopen_s(&fp, fname, "wb");
        if (!fp) {
            fprintf(stderr, "failed to open %s\n", fname);
            MPI_Abort(comm, ierr);
        }
        if (fwrite(&mrcheader, sizeof(MRCheader), 1, fp) != 1) {
            fprintf(stderr, "failed to write an MRC header\n");
            MPI_Abort(comm, ierr);
        }
        imgbuf = (float *)malloc(nx * ny * sizeof(float));
        if (!imgbuf) {
            fprintf(stderr, "failed to allocate image buffer for write\n");
            MPI_Abort(comm, ierr);
        }
    }

    /* writing data */
    for (iz = 0; iz < nz; iz++) {
        if (mypid == 0) {
            /* we write one slice at a time
               copy the local data */
            nyloc = image.ny; // need a reset because it's been overwritten by other processors

            for (ix = 0; ix < nx * nyloc; ix++)
                imgbuf[ix] = image.data[nx * nyloc * iz + ix];

            yoffset = nyloc;

            for (iproc = 1; iproc < nproc; iproc++) {
                // receive a subslice from the ith process
                MPI_Recv(&nyloc, 1, MPI_INT, iproc, iproc, comm, &mpistatus);

                datasize = nx * nyloc;
                MPI_Recv(&imgbuf[nx * yoffset], datasize, MPI_FLOAT, iproc, iproc, comm,
                         &mpistatus);
                yoffset = yoffset + nyloc;
            } // end for iproc
        } else {
            // send data to the 0th process
            if (nproc > 0) {
                datasize = nx * nyloc;
                MPI_Send(&nyloc, 1, MPI_INT, 0, mypid, comm);
                MPI_Send(&(image.data[nx * nyloc * iz]), datasize, MPI_FLOAT, 0, mypid, comm);
            }
        }

        ierr = MPI_Barrier(comm);

        if (mypid == 0) {
            if (fwrite(imgbuf, sizeof(float), nx * ny, fp) != nx * ny) {
                fprintf(stderr, "failed to write mrc data\n");
                MPI_Abort(comm, ierr);
            }
        }
    } // end for iz

    if (mypid == 0) {
        fclose(fp);
        free(imgbuf);
    }
}

void arecWriteImageMergeZ(MPI_Comm comm, const char *fname, arecImage image) {
    /*
     * collect images from all processors and the write out the image stack
     * to a single file. This write merges in the z direction.
     */
    int ierr = 0;
    float amin = 1.0e20, amax = -1.0e20, mean = 0.0;
    FILE *fp = nullptr;
    float *buffer;
    int datasize = 0, nzsum, nzloc, nx, ny, i;
    int mypid, nproc;
    MPI_Status mpistatus;
    MRCheader mrcheader;

    MPI_Comm_rank(comm, &mypid);
    MPI_Comm_size(comm, &nproc);

    nzloc = image.nz;
    nzsum = 0;
    MPI_Allreduce(&nzloc, &nzsum, 1, MPI_INT, MPI_SUM, comm);
    nx = image.nx;
    ny = image.ny;

    mrcheader.nx = nx;
    mrcheader.ny = ny;
    mrcheader.nz = nzsum;
    mrcheader.mode = 2;
    mrcheader.nxstart = 0;
    mrcheader.nystart = 0;
    mrcheader.nzstart = 0;
    mrcheader.mx = nx;
    mrcheader.my = ny;
    mrcheader.mz = nzsum;

    mrcheader.xlen = 1; /* Cell dimensions (Angstroms). */
    mrcheader.ylen = 1; /* Cell dimensions (Angstroms). */
    mrcheader.zlen = 1; /* Cell dimensions (Angstroms). */

    mrcheader.alpha = 90.0; /* Cell angles (Degrees). */
    mrcheader.beta = 90.0;  /* Cell angles (Degrees). */
    mrcheader.gamma = 90.0; /* Cell angles (Degrees). */

    mrcheader.mapc = 1; /* Which axis corresponds to Columns.  */
    mrcheader.mapr = 2; /* Which axis corresponds to Rows.     */
    mrcheader.maps = 3;

    mean = 0.0;
    for (i = 0; i < nx * ny * nzloc; i++) {
        if (image.data[i] > amax) amax = image.data[i];
        if (image.data[i] < amin) amin = image.data[i];
        mean = mean + image.data[i];
    }

    MPI_Allreduce(&mean, &(mrcheader.amean), 1, MPI_FLOAT, MPI_SUM, comm);
    mrcheader.amean = mrcheader.amean / (float)(nx * ny * nzsum);
    MPI_Allreduce(&amax, &(mrcheader.amax), 1, MPI_FLOAT, MPI_MAX, comm);
    MPI_Allreduce(&amin, &(mrcheader.amin), 1, MPI_FLOAT, MPI_MIN, comm);

    mrcheader.ispg = 0;   /* Space group number (0 for images). */
    mrcheader.nsymbt = 0; /* Number of chars used for storing symmetry */
                          /* operators.                                */
    for (i = 0; i < 25; i++)
        mrcheader.user[i] = ' ';
    mrcheader.user[24] = '\0';

    mrcheader.xorigin = 0;
    mrcheader.yorigin = 0;
    mrcheader.zorigin = 0;

    mrcheader.map[0] = 'M';
    mrcheader.map[1] = 'A';
    mrcheader.map[2] = 'P';
    mrcheader.map[3] = '\0';

    mrcheader.rms = 1.0;
    mrcheader.nlabels = 0;

    /* write the header first */
    if (mypid == 0) {
        errno_t err = fopen_s(&fp, fname, "wb");
        if (!fp) {
            fprintf(stderr, "failed to open %s for write\n", fname);
            MPI_Abort(comm, ierr);
        }
        if (fwrite(&mrcheader, sizeof(MRCheader), 1, fp) != 1) {
            fprintf(stderr, "failed to write an MRC header\n");
            MPI_Abort(comm, ierr);
        }
    }

    /* writing data */
    if (mypid == 0) {
        datasize = nx * ny * nzloc;
        if (fwrite(image.data, sizeof(float), datasize, fp) != datasize) {
            fprintf(stderr, "failed to write spider data\n");
            MPI_Abort(comm, ierr);
        }
        buffer = (float *)malloc(datasize * sizeof(float));
        for (i = 1; i < nproc; i++) {
            /* receive data from the ith process */
            MPI_Recv(&nzloc, 1, MPI_INT, i, i, comm, &mpistatus);
            datasize = nx * ny * nzloc;
            MPI_Recv(buffer, datasize, MPI_FLOAT, i, i, comm, &mpistatus);
            if (fwrite(buffer, sizeof(float), datasize, fp) != datasize) {
                fprintf(stderr, "failed to write image data\n");
                MPI_Abort(comm, ierr);
            }
        } /* end for i */
        free(buffer);
    } else {
        /* send data to the 0th process */
        MPI_Send(&nzloc, 1, MPI_INT, 0, mypid, comm);
        datasize = nx * ny * nzloc;
        MPI_Send(image.data, datasize, MPI_FLOAT, 0, mypid, comm);
    }

    if (mypid == 0) fclose(fp);
}
