 
CC := g++ # This is the main compiler
# CC := clang --analyze # and comment out the linker last line for sanity

TARGET ?= ./bin/prealign
SRC_DIRS ?= ./Prealign_2017/

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s)
OBJS := $(addsuffix .o,$(basename $(SRCS)))
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CFLAGS = -O3 -fopenmp
CPPFLAGS ?= $(INC_MPI) $(INC_FLAGS) -MMD -MP -std=c++1y -g -Wall -Wno-unknown-pragmas -O3 -fopenmp

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $@ $(LOADLIBES) $(LDLIBS)

prealign: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LIBRARY) -o $(TARGET)

.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS) $(DEPS)

-include $(DEPS) 
