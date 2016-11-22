# User defined parameters
KERNELS 	  	= omp3
COMPILER    	= INTEL
CFLAGS_INTEL	= -O3 -g -qopenmp -no-prec-div -xhost -std=c99
OPTIONS		  	= -DDEBUG

# Default compiler
MAPP_COMPILER = mpicc
MAPP_LINKER	  = mpicc
MAPP_FLAGS	  = $(CFLAGS_$(COMPILER))
MAPP_LDFLAGS  = 

OBJS  = $(patsubst %.c, obj/$(KERNELS)/%.o, $(wildcard *.c))
OBJS += $(patsubst %.c, obj/$(KERNELS)/%.o, $(wildcard ../*.c))

hot: make_build_dir $(OBJS) Makefile
	$(MAPP_LINKER) $(MAPP_FLAGS) $(OBJS) $(MAPP_LDFLAGS) -o hot.exe

# Rule to make controlling code
obj/$(KERNELS)/%.o: %.c Makefile 
	$(MAPP_COMPILER) $(MAPP_FLAGS) $(OPTIONS) -c $< -o $@

make_build_dir:
	@mkdir -p obj/
	@mkdir -p obj/$(KERNELS)

clean:
	rm -rf obj/* hot.exe *.vtk

