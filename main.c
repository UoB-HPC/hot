#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "main.h"

#define ind0 (ii*nx + jj)
#define ind1 (ii*(nx+1) + jj)

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

  // Fetch the user's parameters
  printf("Using the CG solver\n");

  solve(
      mesh.local_nx, mesh.local_ny, mesh.dt, mesh.niters, 
      state.x, state.r, state.p, state.rho, state.s_x, state.s_y, 
      state.Ap, state.e, mesh.edgedx, mesh.edgedy);
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
  mesh->edgedx = (int*)malloc(sizeof(int)*mesh->local_nx);
  mesh->edgedy = (int*)malloc(sizeof(int)*mesh->local_ny);

#pragma omp parallel for
  for(int ii = 0; ii < mesh->local_nx; ++ii) {
    mesh->edgedx[ii] = mesh->width/mesh->global_nx;
  }
#pragma omp parallel for
  for(int ii = 0; ii < mesh->local_ny; ++ii) {
    mesh->edgedy[ii] = mesh->width/mesh->global_ny;
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
  state->e = (double*)malloc(sizeof(double)*nx*ny);
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
      state->e[ind0] = 0.0;
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
      if((ii > ny/8 && ii <= ny/4 && jj >= 0 && jj <= nx/2+nx/16) || 
          (ii >= ny/4 && ii <= 7*ny/8 && jj >= nx/2-nx/16 && jj <= nx/2+nx/16) ||
          (ii >= 3*ny/4 && ii <= 7*ny/8 && jj >= nx/2-nx/16 && jj < nx)) {
        state->rho[ind0] = 1.0;
      }
      else {
        state->rho[ind0] = 100.0;
        state->e[ind0] = 0.0001;
      }
    }
  }

  write_to_visit(nx, ny, 0, 0, state->rho, "initial_crooked", 0, 0.0);

  // Homogenous slab problem 
#pragma omp parallel for
  for(int ii = 0; ii < ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < nx; ++jj) {
      state->rho[ind0] = 1.0;
      state->e[ind0] = 1.0;
    }
  }
}

// Performs the CG solve
void solve(
    const int nx, const int ny, const double dt, const int niters, double* x, 
    double* r, double* p, const double* rho, double* s_x, double* s_y, 
    double* Ap, double* e, const int* edgedx, const int* edgedy)
{
  initialise_cg(nx, ny, dt, p, rho, s_x, s_y, Ap, edgedx, edgedy);

#ifdef DEBUG
  printf("\nVector Ax initial: \n");
  print_vec(nx, nx, Ap);
#endif

  // Store initial residual
  double old_rr = 0.0;
  store_residual(nx, ny, e, Ap, r, &old_rr);

#ifdef DEBUG
  printf("\nresidual: \n");
  print_vec(nx, ny, r);
#endif

  // Store initial conjugate
  copy_vec(nx, ny, r, p);

  for(int ii = 0; ii < niters; ++ii) {
    double alpha = calculate_alpha(nx, ny, s_x, s_y, old_rr, p, Ap); 

#ifdef DEBUG
    printf("alpha: %.12e\n", alpha);
#endif

    double new_rr = 0.0;
    double beta = calculate_beta(nx, ny, alpha, old_rr, x, p, r, Ap, &new_rr);

#ifdef DEBUG
    printf("beta: %.12e\n", beta);
#endif

    // Check if the solution has converged
    if(fabs(new_rr) < 1.0e-05) {
      printf("exiting at iteration %d with new_rr: %.12e\n", ii, new_rr);
      break;
    }

    update_conjugate(nx, ny, beta, p, r);

    // Store the old squared residual
    old_rr = new_rr;

#ifdef DEBUG
    printf("old_rr: %.12e\n", old_rr);
#endif
  }

  printf("\nResult: \n");
  print_vec(nx, ny, x);
}

// Initialises the CG solver
void initialise_cg(
    const int nx, const int ny, const double dt, double* p, const double* rho, 
    double* s_x, double* s_y, double* Ap, const int* edgedx, const int* edgedy)
{
  // https://inldigitallibrary.inl.gov/sti/3952796.pdf
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      s_x[ind1] = (dt*CONDUCTIVITY*(rho[ind0]+rho[ind0-1]))/
        (2.0*(rho[ind0]*rho[ind0-1])*(edgedx[ii]*edgedx[ii]));
      s_y[ind1] = (dt*CONDUCTIVITY*(rho[ind0]+rho[ind0-ny]))/
        (2.0*(rho[ind0]*rho[ind0-ny])*(edgedy[ii]*edgedy[ii]));
    }
  }

  // Initialise the guess at solution vector
  // You don't need to use a matrix as the band matrix is fully predictable
  // from the 5pt stencil
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      Ap[ind0] = 
        - s_y[ind1-nx]*p[ind0-nx]
        - s_x[ind1-1]*p[ind0-1] 
        + (1.0+s_x[ind1-1]+s_x[ind1]+s_y[ind1-nx]+s_y[ind1])*p[ind0]
        - s_x[ind1]*p[ind0+1]
        - s_y[ind1]*p[ind0+nx];
    }
  }
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

// Calculates a value for alpha
double calculate_alpha(
    const int nx, const int ny, const double* s_x, const double* s_y,
    double old_rr, double* p, double* Ap)
{
  double pAp = 0.0;

#pragma omp parallel for reduction(+: pAp)
  for(int ii = PAD; ii < ny - PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx - PAD; ++jj) {
      Ap[ind0] = 
        - s_y[ind1-nx]*p[ind0-nx]
        - s_x[ind1-1]*p[ind0-1] 
        + (1.0+s_x[ind1-1]+s_x[ind1]+s_y[ind1-nx]+s_y[ind1])*p[ind0]
        - s_x[ind1]*p[ind0+1]
        - s_y[ind1]*p[ind0+nx];
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

// Update the residual at the current step
void store_residual(
    int nx, int ny, double* e, double* Ap, double* r, double* old_rr)
{
  double rr_temp = 0.0;

#pragma omp parallel for reduction(+: rr_temp)
  for(int ii = 0; ii < ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < nx; ++jj) {
      r[ind0] = e[ind0] - Ap[ind0];
      rr_temp += r[ind0]*r[ind0];
    }
  }

  *old_rr = rr_temp;
}

// Copies the vector src into dest
void copy_vec(
    const int nx, const int ny, double* src, double* dest)
{
#pragma omp parallel for simd
  for(int ii = 0; ii < nx*ny; ++ii) {
    dest[ii] = src[ii];
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

