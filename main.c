#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "profiler.h"
#include "main.h"

#define ind0 (ii*nx + jj)
#define ind1 (ii*(nx+1) + jj)

struct Profile compute_profile = {0};

int main(int argc, char** argv)
{
  if(argc < 4)
  {
    printf("Usage: ./hot.exe <nx> <ny> <niters>\n");
    exit(1);
  }

  Mesh mesh = {0};
  State state = {0};
  mesh.global_nx = atoi(argv[1]);
  mesh.global_ny = atoi(argv[2]);
  mesh.local_nx = atoi(argv[1]) + 2*PAD;
  mesh.local_ny = atoi(argv[2]) + 2*PAD;
  mesh.width = WIDTH;
  mesh.height = HEIGHT;
  mesh.dt = MAX_DT;
  mesh.rank = MASTER;
  mesh.nranks = 1;
  mesh.niters = atoi(argv[3]);

  initialise_mesh(&mesh);
  initialise_state(mesh.local_nx, mesh.local_ny, &state);
  initialise_comms(argc, argv, &mesh);

  struct Profile wallclock = {0};

  int tt = 0;
  double elapsed_sim_time = 0.0;
  for(tt = 0; tt < 1; ++tt) {
    START_PROFILING(&wallclock);
    solve(
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt, mesh.niters, state.x, 
        state.r, state.p, state.rho, state.s_x, state.s_y, 
        state.Ap, mesh.edgedx, mesh.edgedy);
    STOP_PROFILING(&wallclock, "wallclock");

    elapsed_sim_time += mesh.dt;
  }

  double global_wallclock = 0.0;
  if(tt > 0) {
#ifdef MPI
    struct ProfileEntry pe = profiler_get_profile_entry(&wallclock, "wallclock");
    MPI_Reduce(&pe.time, &global_wallclock, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
#endif
  }

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profile);
    printf("Wallclock %.2fs, Elapsed Simulation Time %.4fs\n", global_wallclock, elapsed_sim_time);
  }

  write_to_visit(
      mesh.local_nx, mesh.local_ny, mesh.x_off, mesh.y_off, 
      state.x, "final_result", 1, mesh.dt);
}

// This is currently duplicated from the hydro package
void initialise_comms(
    int argc, char** argv, Mesh* mesh)
{
  for(int ii = 0; ii < NNEIGHBOURS; ++ii) {
    mesh->neighbours[ii] = EDGE;
  }

#ifdef MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mesh->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mesh->nranks);

  decompose_2d_cartesian(
      mesh->rank,  mesh->nranks,  mesh->global_nx,  mesh->global_ny, 
      mesh->neighbours, &mesh->local_nx, &mesh->local_ny, &mesh->x_off, &mesh->y_off);

  // Add on the halo padding to the local mesh
  mesh->local_nx += 2*PAD;
  mesh->local_ny += 2*PAD;
#endif 

  if(mesh->rank == MASTER)
    printf("Problem dimensions %dx%d for %d iterations.\n", 
        mesh->global_nx, mesh->global_ny, mesh->niters);
}

// Initialises the mesh
void initialise_mesh(Mesh* mesh) 
{
  mesh->edgedx = (double*)malloc(sizeof(double)*(mesh->local_nx+1));
  mesh->edgedy = (double*)malloc(sizeof(double)*(mesh->local_ny+1));

#pragma omp parallel for
  for(int ii = 0; ii < mesh->local_nx+1; ++ii) {
    mesh->edgedx[ii] = (double)mesh->width/mesh->global_nx;
  }
#pragma omp parallel for
  for(int ii = 0; ii < mesh->local_ny+1; ++ii) {
    mesh->edgedy[ii] = (double)mesh->width/mesh->global_ny;
  }

  mesh->north_buffer_out 
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->east_buffer_out  
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  mesh->south_buffer_out 
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->west_buffer_out  
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  mesh->north_buffer_in  
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->east_buffer_in   
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  mesh->south_buffer_in  
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->west_buffer_in   
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
}

// Initialises the state variables
void initialise_state(const int nx, const int ny, State* state) 
{
  state->Ap = (double*)malloc(sizeof(double)*nx*ny);
  state->r = (double*)malloc(sizeof(double)*nx*ny);
  state->x = (double*)malloc(sizeof(double)*nx*ny);
  state->p = (double*)malloc(sizeof(double)*nx*ny);
  state->s_x = (double*)malloc(sizeof(double)*(nx+1)*(ny+1));
  state->s_y = (double*)malloc(sizeof(double)*(nx+1)*(ny+1));
  state->rho = (double*)malloc(sizeof(double)*nx*ny);

#pragma omp parallel for
  for(int ii = 0; ii < ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < nx; ++jj) {
      state->x[ind0] = 0.0;
      state->r[ind0] = 0.0;
      state->p[ind0] = 0.0;
      state->Ap[ind0] = 0.0;
      state->rho[ind0] = 0.0;
    }
  }
#pragma omp parallel for
  for(int ii = 0; ii < (ny+1); ++ii) {
#pragma omp simd
    for(int jj = 0; jj < (nx+1); ++jj) {
      state->s_x[ind1] = 0.0;
      state->s_y[ind1] = 0.0;
    }
  }

  // Crooked pipe problem
#pragma omp parallel for
  for(int ii = 0; ii < ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < nx; ++jj) {
      if((ii >= ny/4 && ii <= 7*ny/8 && jj >= nx/2-nx/16 && jj <= nx/2+nx/16) ||
          (ii >= 3*ny/4 && ii <= 7*ny/8 && jj >= nx/2-nx/16 && jj < nx) ||
          (ii > ny/8 && ii <= ny/4 && jj >= 0 && jj <= nx/2+nx/16)) {
        state->rho[ind0] = 0.1;
        state->x[ind0] = 0.1*state->rho[ind0];
      }
      else {
        state->rho[ind0] = 1.0e3;
        state->x[ind0] = 1.0e-5*state->rho[ind0];
      }

      // Heat a region
      if (ii > ny/8 && ii <= ny/4 && jj >= 10 && jj <= nx/8) {
        state->rho[ind0] = 0.1;
        state->x[ind0] = 1.0e3*state->rho[ind0];
      }
    }
  }

  write_to_visit(nx, ny, 0, 0, state->x, "initial_crooked", 0, 0.0);
}

// Performs the CG solve
void solve(
    const int nx, const int ny, Mesh* mesh, const double dt, const int niters, double* x, 
    double* r, double* p, const double* rho, double* s_x, double* s_y, 
    double* Ap, const double* edgedx, const double* edgedy)
{
  // Store initial residual
  double old_rr = 0.0;
  START_PROFILING(&compute_profile);
  initialise_cg(nx, ny, dt, p, r, x, rho, s_x, s_y, &old_rr, edgedx, edgedy);
  STOP_PROFILING(&compute_profile, "initialise cg");

  handle_boundary(nx, ny, mesh, p, PACK);
  handle_boundary(nx, ny, mesh, x, PACK);

  for(int ii = 0; ii < niters; ++ii) {
    START_PROFILING(&compute_profile);
    double alpha = calculate_alpha(nx, ny, s_x, s_y, old_rr, p, Ap);
    STOP_PROFILING(&compute_profile, "calculate alpha");

    double new_rr = 0.0;
    START_PROFILING(&compute_profile);
    double beta = calculate_beta(nx, ny, alpha, old_rr, x, p, r, Ap, &new_rr);
    STOP_PROFILING(&compute_profile, "calculate beta");

    // Check if the solution has converged
    if(fabs(new_rr) < 1.0e-05) {
      printf("exiting at iteration %d with new_rr: %.12e\n", ii, new_rr);
      break;
    }

    START_PROFILING(&compute_profile);
    update_conjugate(nx, ny, beta, r, p);
    STOP_PROFILING(&compute_profile, "update conjugate");

    handle_boundary(nx, ny, mesh, p, PACK);
    handle_boundary(nx, ny, mesh, x, PACK);

    // Store the old squared residual
    old_rr = new_rr;
  }
}

// Initialises the CG solver
void initialise_cg(
    const int nx, const int ny, const double dt, double* p, double* r,
    const double* x, const double* rho, double* s_x, double* s_y, double* initial_rr, 
    const double* edgedx, const double* edgedy)
{
  // https://inldigitallibrary.inl.gov/sti/3952796.pdf
  // Take the average of the coefficients at the cells surrounding 
  // each face
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      s_x[ind1] = (dt*CONDUCTIVITY*(rho[ind0]+rho[ind0-1]))/
        (2.0*rho[ind0]*rho[ind0-1]*edgedx[jj]*edgedx[jj]*HEAT_CAPACITY);
      s_y[ind1] = (dt*CONDUCTIVITY*(rho[ind0]+rho[ind0-nx]))/
        (2.0*rho[ind0]*rho[ind0-nx]*edgedy[ii]*edgedy[ii]*HEAT_CAPACITY);
    }
  }

  double rr = 0.0;
#pragma omp parallel for reduction(+: rr)
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      r[ind0] = x[ind0] -
        ((1.0+s_y[ind1]+s_x[ind1]+s_x[ind1+1]+s_y[ind1+(nx+1)])*x[ind0]
         - s_y[ind1]*x[ind0-nx]
         - s_x[ind1]*x[ind0-1] 
         - s_x[ind1+1]*x[ind0+1]
         - s_y[ind1+(nx+1)]*x[ind0+nx]);
      p[ind0] = r[ind0];
      rr += r[ind0]*r[ind0];
    }
  }
  *initial_rr = rr;
}

// Calculates a value for alpha
double calculate_alpha(
    const int nx, const int ny, const double* s_x, const double* s_y,
    double old_rr, double* p, double* Ap)
{
  double pAp = 0.0;

  // You don't need to use a matrix as the band matrix is fully predictable
  // from the 5pt stencil
#pragma omp parallel for reduction(+: pAp)
  for(int ii = PAD; ii < ny - PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx - PAD; ++jj) {
      Ap[ind0] = 
        (1.0+s_y[ind1]+s_x[ind1]+s_x[ind1+1]+s_y[ind1+(nx+1)])*p[ind0]
        - s_y[ind1]*p[ind0-nx]
        - s_x[ind1]*p[ind0-1] 
        - s_x[ind1+1]*p[ind0+1]
        - s_y[ind1+(nx+1)]*p[ind0+nx];
      pAp += p[ind0]*Ap[ind0];
    }
  }

  return old_rr / pAp;
}

// Updates the current guess using the calculated alpha
double calculate_beta(
    int nx, int ny, double alpha, double old_rr, 
    double* x, double* p, double* r, double* Ap, double* new_rr)
{
  double rr_temp = 0.0;

#pragma omp parallel for reduction(+: rr_temp)
  for(int ii = PAD; ii < ny - PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx - PAD; ++jj) {
      x[ind0] += alpha*p[ind0];
      r[ind0] -= alpha*Ap[ind0];
      rr_temp += r[ind0]*r[ind0];
    }
  }

  *new_rr = rr_temp;
  return rr_temp / old_rr;
}

// Updates the conjugate from the calculated beta and residual
void update_conjugate(
    const int nx, const int ny, const double beta, const double* r, double* p)
{
#pragma omp parallel for
  for(int ii = PAD; ii < ny - PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx - PAD; ++jj) {
      p[ind0] = r[ind0] + beta*p[ind0];
    }
  }
}

// Prints the vector to std out
void print_vec(
    const int nx, const int ny, double* a)
{
  for(int ii = 0; ii < ny; ++ii) {
    for(int jj = 0; jj < nx; ++jj) {
      printf("%.3e ", a[ii*nx+jj]);
    }
    printf("\n");
  }
}

// Enforce reflective boundary conditions on the problem state
void handle_boundary(
    const int nx, const int ny, Mesh* mesh, double* arr, const int pack)
{
  START_PROFILING(&compute_profile);

#ifdef MPI
  int nmessages = 0;
  MPI_Request out_req[NNEIGHBOURS];
  MPI_Request in_req[NNEIGHBOURS];
#endif

  if(mesh->neighbours[WEST] == EDGE) {
    // reflect at the west
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        arr[ii*nx + (PAD - 1 - dd)] = arr[ii*nx + (PAD + dd)];
      }
    }
  }

#ifdef MPI
  else if(pack) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        mesh->west_buffer_out[ii*PAD+dd] = arr[(ii*nx)+(PAD+dd)];
      }
    }

    MPI_Isend(mesh->west_buffer_out, ny*PAD, MPI_DOUBLE,
        mesh->neighbours[WEST], 3, MPI_COMM_WORLD, &out_req[WEST]);
    MPI_Irecv(mesh->west_buffer_in, ny*PAD, MPI_DOUBLE, 
        mesh->neighbours[WEST], 2, MPI_COMM_WORLD, &in_req[nmessages++]);
  }
#endif

  // Reflect at the east
  if(mesh->neighbours[EAST] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        arr[ii*nx + (nx - PAD + dd)] = arr[ii*nx + (nx - 1 - PAD - dd)];
      }
    }
  }
#ifdef MPI
  else if(pack) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        mesh->east_buffer_out[ii*PAD+dd] = arr[(ii*nx)+(nx-2*PAD+dd)];
      }
    }

    MPI_Isend(mesh->east_buffer_out, ny*PAD, MPI_DOUBLE, 
        mesh->neighbours[EAST], 2, MPI_COMM_WORLD, &out_req[EAST]);
    MPI_Irecv(mesh->east_buffer_in, ny*PAD, MPI_DOUBLE,
        mesh->neighbours[EAST], 3, MPI_COMM_WORLD, &in_req[nmessages++]);
  }
#endif

  // Reflect at the north
  if(mesh->neighbours[NORTH] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        arr[(ny - PAD + dd)*nx + jj] = arr[(ny - 1 - PAD - dd)*nx + jj];
      }
    }
  }
#ifdef MPI
  else if(pack) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        mesh->north_buffer_out[dd*nx+jj] = arr[(ny-2*PAD+dd)*nx+jj];
      }
    }

    MPI_Isend(mesh->north_buffer_out, nx*PAD, MPI_DOUBLE, 
        mesh->neighbours[NORTH], 1, MPI_COMM_WORLD, &out_req[NORTH]);
    MPI_Irecv(mesh->north_buffer_in, nx*PAD, MPI_DOUBLE,
        mesh->neighbours[NORTH], 0, MPI_COMM_WORLD, &in_req[nmessages++]);
  }
#endif

  // reflect at the south
  if(mesh->neighbours[SOUTH] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        arr[(PAD - 1 - dd)*nx + jj] = arr[(PAD + dd)*nx + jj];
      }
    }
  }
#ifdef MPI
  else if (pack) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        mesh->south_buffer_out[dd*nx+jj] = arr[(PAD+dd)*nx+jj];
      }
    }

    MPI_Isend(mesh->south_buffer_out, nx*PAD, MPI_DOUBLE, 
        mesh->neighbours[SOUTH], 0, MPI_COMM_WORLD, &out_req[SOUTH]);
    MPI_Irecv(mesh->south_buffer_in, nx*PAD, MPI_DOUBLE,
        mesh->neighbours[SOUTH], 1, MPI_COMM_WORLD, &in_req[nmessages++]);
  }
#endif

  // Unpack the buffers
#ifdef MPI
  if(pack) {
    MPI_Waitall(nmessages, in_req, MPI_STATUSES_IGNORE);

    if(mesh->neighbours[NORTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = 0; jj < nx; ++jj) {
          arr[(ny-PAD+dd)*nx+jj] = mesh->north_buffer_in[dd*nx+jj];
        }
      }
    }

    if(mesh->neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = 0; jj < nx; ++jj) {
          arr[dd*nx + jj] = mesh->south_buffer_in[dd*nx+jj];
        }
      }
    }

    if(mesh->neighbours[WEST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < ny; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          arr[ii*nx + dd] = mesh->west_buffer_in[ii*PAD+dd];
        }
      }
    }

    if(mesh->neighbours[EAST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < ny; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          arr[ii*nx + (nx-PAD+dd)] = mesh->east_buffer_in[ii*PAD+dd];
        }
      }
    }
  }
#endif

  STOP_PROFILING(&compute_profile, "communications");
}

