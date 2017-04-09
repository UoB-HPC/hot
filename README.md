# hot
A heat diffusion mini-app that uses a CG solver

# Purpose

This application is a simplified heat diffusion application, that uses a CG solver without a pre-conditioner. The intention is for this application to be used to perform performance evaluation on modern architecture.

# Build

Before building the dependent `hot` application, it is necessary to clone the application into the `arch` project. The instructions can be found on the `arch` project README.

```
git clone git@github.com:uob-hpc/arch
cd arch
git clone git@github.com:uob-hpc/hot
cd hot
```

The `hot` build process is intended to be simple, and has been tested on a number of platforms.

```
make KERNELS=omp3 COMPILER=INTEL
```

The `KERNELS` option determines the particular kernel set that will be used when building the project. At this time the difference between the kernel sets is that they are written with different programming models. When you clone the repository there are multiple directories that contain duplicates of the core computational kernels, ported to different models and the name of the directory is the value to be used with the KERNELS option.

A number of other switches and options are provided:

- `DEBUG=<yes/no>` - 'yes' switches off optimisation and adds debug flags
- `MPI=<yes/no>` - 'yes' turns off any use of MPI within the application.
- `DECOMP=<TILES/ROWS/COLS> - determines the decomposition strategy (Warning: this hasn't been very well tested yet).
- The `OPTIONS` makefile variable is used to allow visit dumps, with `-DVISIT_DUMP`, and profiling, with `-DENABLE_PROFILING`.

# Run

Upon building, an binary file will be output with the extension of the value of KERNELS. e.g. `hot.omp3`. You can run the application with:

```
./hot.omp3 hot.params
```

# Configuration Files

The configuration files expose a number of key parameters for the application.

- `iterations` - the number of outer timestep iterations the application will proceed through
- `max_inners` - the maximum number of iterations allowed before convergence is abandoned
- `dt` - the timestep for the application
- `nx` - the number of cells in the x-dimension
- `ny` - the number of cells in the y-dimension
- `visit_dump` - whether the application should output visit dumps of the result

TODO: Describe the `problem` and `source` descriptions in the parameter file.
