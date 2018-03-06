# This Makefile can be used on franklin,hopper,specter
# For other machines, make sure the following libraries are
# properly installed and linked
#  HDF5

MACHINETYPE = home

DEFINE = -DUSE_MPI

###################################################
#                On home
###################################################
ifeq ($(MACHINETYPE),home)


MPICXX = mpic++
CC     = mpic++
CFLAGS = -O3
DEFINE = -DUSE_MPI

MPIINC = /usr/include/mpich/
FFTLIB = -L/usr/lib/x86_64-linux-gnu/ -lfftw3f

endif


#INCLUDE = -I$(HDFINC) -I$(FFTINC) -I.
INCLUDE = -I$(FFTINC) -I.
#LIBRARY = $(HDFLIB) $(FFTLIB) -lm
LIBRARY = $(FFTLIB) -lm

FLAGS = $(INCLUDE) $(LIBRARY) 

.SUFFIXES:
.SUFFIXES:	.cpp	.o
.SUFFIXES:	.c	.o

.cpp.o:
	$(CC) $(CFLAGS) $(DEFINE) $(INCLUDE) -c $<
.c.o:
	$(CC) $(CFLAGS) $(DEFINE) $(INCLUDE) -c $<

OBJS = arecImage.o arecImageIO.o arecImageIO_mpi.o align2dstack.o arecsirt.o \
       arecproject.o arec3dutil.o areccgls.o edgeslope.o ReadAndDistAngles.o \
       arecsirt_KB.o areccgls_KB.o imagetools.o

arec3d: arec3d.o $(OBJS)
	$(CC) $(CFLAGS) arec3d.o $(OBJS) $(LIBRARY) -o arec3d



clean:
	rm -rf *~ *.o imagedriver runsirt runcylsirt runcylsirt2 runali3dcyl runalistack imageiotest runarec3d runrotalign

