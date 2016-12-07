# User defined parameters
KERNELS 	  	= omp3
COMPILER    	= INTEL
CFLAGS_INTEL	= -O3 -g -qopenmp -no-prec-div -xhost -std=gnu99 
CFLAGS_CRAY	  = -hlist=a -lrt
OPTIONS		  	= -DENABLE_PROFILING -DDEBUG #-DMPI

# Default compiler
MULTI_COMPILER  = icc
MULTI_LINKER    = $(MULTI_COMPILER)
MULTI_FLAGS     = $(CFLAGS_$(COMPILER))
MULTI_LDFLAGS   =
MULTI_BUILD_DIR = ../obj
MULTI_DIR       = ..

SRC  			 = $(wildcard *.c)
SRC  			+= $(wildcard $(KERNELS)/*.c)
SRC  			+= $(wildcard $(MULTI_DIR)/$(KERNELS)/*.c)
SRC 			+= $(subst main.c,, $(wildcard $(MULTI_DIR)/*.c))
SRC_CLEAN  = $(subst $(MULTI_DIR)/,,$(SRC))
OBJS 			 = $(patsubst %.c, $(MULTI_BUILD_DIR)/%.o, $(SRC_CLEAN))

hot: make_build_dir $(OBJS) Makefile
	$(MULTI_LINKER) $(MULTI_FLAGS) $(OBJS) $(MULTI_LDFLAGS) -o hot.exe

# Rule to make controlling code
$(MULTI_BUILD_DIR)/%.o: %.c Makefile 
	$(MULTI_COMPILER) $(MULTI_FLAGS) $(OPTIONS) -c $< -o $@

$(MULTI_BUILD_DIR)/%.o: $(MULTI_DIR)/%.c Makefile 
	$(MULTI_COMPILER) $(MULTI_FLAGS) $(OPTIONS) -c $< -o $@

make_build_dir:
	@mkdir -p $(MULTI_BUILD_DIR)/
	@mkdir -p $(MULTI_BUILD_DIR)/$(KERNELS)

clean:
	rm -rf $(MULTI_BUILD_DIR)/* hot.exe *.vtk *.bov *.dat *.optrpt *.cub *.ptx

