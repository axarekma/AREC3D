


CC := mpic++ # This is the main compiler
# CC := clang --analyze # and comment out the linker last line for sanity

TARGET ?= ./bin/arec3d
SRC_DIRS ?= ./AREC3D/

DEFINE = -DUSE_MPI
INC_MPI = -I /usr/include/mpich/
FFTLIB ?= -L/usr/lib/x86_64-linux-gnu/ -lfftw3f 

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s)
OBJS := $(addsuffix .o,$(basename $(SRCS)))
DEPS := $(OBJS:.o=.d)

LIBRARY =  $(FFTLIB) -lm
INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CFLAGS = -O3 -DNDEBUG -march=native
CPPFLAGS ?= $(INC_MPI) $(INC_FLAGS) -MMD -MP -std=c++11 -g -Wunreachable-code -Wshadow -Wextra -Wall -Wno-unknown-pragmas -O3 -DNDEBUG -march=native -fopenmp


AREC3D: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LIBRARY) -o $(TARGET)

.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS) $(DEPS)

-include $(DEPS) 
