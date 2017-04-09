# User defined parameters
KERNELS 	  	= omp3
COMPILER    	= INTEL
MPI						= yes
DECOMP				= TILES
CFLAGS_INTEL	= -qopenmp -no-prec-div -std=gnu99 -DINTEL \
								-Wall -qopt-report=5 -xhost
CFLAGS_GCC		= -std=gnu99 -fopenmp -march=native -Wall #-std=gnu99
CFLAGS_CRAY		= -lrt -hlist=a
OPTIONS		  	= -DENABLE_PROFILING 

ifeq ($(DEBUG), yes)
  OPTIONS += -O0 -g -DDEBUG 
else
  OPTIONS += -O3
endif

ifeq ($(MPI), yes)
  OPTIONS += -DMPI
endif

ifeq ($(DECOMP), TILES)
OPTIONS += -DTILES
endif
ifeq ($(DECOMP), ROWS)
OPTIONS += -DROWS
endif
ifeq ($(DECOMP), COLS)
OPTIONS += -DCOLS
endif

# Default compiler
ARCH_COMPILER_CC   = mpicc
ARCH_COMPILER_CPP  = mpic++
ARCH_LINKER    		= $(ARCH_COMPILER_CC)
ARCH_FLAGS     		= $(CFLAGS_$(COMPILER))
ARCH_LDFLAGS   		= $(ARCH_FLAGS) -lm
ARCH_BUILD_DIR 		= ../obj
ARCH_DIR       		= ..

ifeq ($(KERNELS), cuda)
include Makefile.cuda
endif

# Get specialised kernels
SRC  			 = $(wildcard *.c)
SRC  			+= $(wildcard $(KERNELS)/*.c)
SRC  			+= $(wildcard $(ARCH_DIR)/$(KERNELS)/*.c)
SRC 			+= $(subst main.c,, $(wildcard $(ARCH_DIR)/*.c))
SRC_CLEAN  = $(subst $(ARCH_DIR)/,,$(SRC))
OBJS 			+= $(patsubst %.c, $(ARCH_BUILD_DIR)/%.o, $(SRC_CLEAN))

hot: make_build_dir $(OBJS) Makefile
	$(ARCH_LINKER) $(OBJS) $(ARCH_LDFLAGS) -o hot.$(KERNELS)

# Rule to make controlling code
$(ARCH_BUILD_DIR)/%.o: %.c Makefile 
	$(ARCH_COMPILER_CC) $(ARCH_FLAGS) $(OPTIONS) -c $< -o $@

$(ARCH_BUILD_DIR)/%.o: $(ARCH_DIR)/%.c Makefile 
	$(ARCH_COMPILER_CC) $(ARCH_FLAGS) $(OPTIONS) -c $< -o $@

make_build_dir:
	@mkdir -p $(ARCH_BUILD_DIR)/
	@mkdir -p $(ARCH_BUILD_DIR)/$(KERNELS)

clean:
	rm -rf $(ARCH_BUILD_DIR)/* hot.$(KERNELS) *.vtk *.bov *.dat *.optrpt *.cub *.ptx

