# User defined parameters
KERNELS 	  			 = omp3
COMPILER    			 = INTEL
MPI								 = no
DECOMP						 = TILES
ARCH_COMPILER_CC   = icc
ARCH_COMPILER_CPP  = icpc
OPTIONS		  			 = -DENABLE_PROFILING 

# Compiler-specific flags
CFLAGS_INTEL			 = -qopenmp -no-prec-div -std=c11 -DINTEL \
										 -Wall -qopt-report=5 -xCORE-AVX512 -qopt-zmm-usage=high
CFLAGS_INTEL_KNL	 = -O3 -qopenmp -no-prec-div -std=c11 -DINTEL \
										 -xMIC-AVX512 -Wall -qopt-report=5
CFLAGS_GCC				 = -std=c11 -fopenmp -march=native -Wall
CFLAGS_GCC_KNL   	 = -O3 -fopenmp -std=c11 \
										 -mavx512f -mavx512cd -mavx512er -mavx512pf #-fopt-info-vec-all
CFLAGS_GCC_POWER   = -O3 -mcpu=power8 -mtune=power8 -fopenmp -std=c11
CFLAGS_CRAY				 = -lrt -hlist=a
CFLAGS_XL					 = -O3 -qsmp=omp
CFLAGS_XL_OMP4		 = -qsmp -qoffload
CFLAGS_CLANG_OMP4  = -O3 -Wall -fopenmp-targets=nvptx64-nvidia-cuda \
										 -fopenmp-nonaliased-maps -fopenmp=libomp \
										 --cuda-path=$(CUDA_PATH) -DCLANG
CFLAGS_CLANG			 = -std=c11 -fopenmp=libiomp5 -march=native -Wall
CFLAGS_PGI				 = -O3 -fast -mp

ifeq ($(KERNELS), cuda)
  CHECK_CUDA_ROOT = yes
endif
ifeq ($(COMPILER), CLANG_OMP4)
  CHECK_CUDA_ROOT = yes
endif

ifeq ($(CHECK_CUDA_ROOT), yes)
ifeq ("${CUDA_PATH}", "")
$(error "$$CUDA_PATH is not set, please set this to the root of your CUDA install.")
endif
endif

ifeq ($(DEBUG), yes)
  OPTIONS += -O0 -g -DDEBUG 
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
ARCH_LINKER    		= $(ARCH_COMPILER_CC)
ARCH_FLAGS     		= $(CFLAGS_$(COMPILER))
ARCH_LDFLAGS   		= $(ARCH_FLAGS) -lm -ldl
ARCH_BUILD_DIR 		= ../obj/hot/
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

