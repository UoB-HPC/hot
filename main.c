#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "main.h"

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
  mesh.width = 10.0;
  mesh.height = 10.0;
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
      state.Ap, state.b, mesh.celldx, mesh.celldy);
}

// Initialises the mesh
void initialise_mesh(Mesh* mesh) 
{
#pragma omp parallel for
  for(int ii = 0; ii < mesh->local_nx; ++ii) {
    mesh->celldx[ii] = mesh->width/mesh->global_nx;
  }
#pragma omp parallel for
  for(int ii = 0; ii < mesh->local_ny; ++ii) {
    mesh->celldy[ii] = mesh->width/mesh->global_ny;
  }
}

// Initialises the state variables
void initialise_state(const int nx, const int ny, State* state) 
{
  state->Ap = (double*)malloc(sizeof(double)*nx*ny);
  state->b = (double*)malloc(sizeof(double)*nx*ny);
  state->r = (double*)malloc(sizeof(double)*nx*ny);
  state->x = (double*)malloc(sizeof(double)*nx*ny);
  state->p = (double*)malloc(sizeof(double)*nx*ny);
  state->s_x = (double*)malloc(sizeof(double)*nx*ny);
  state->s_y = (double*)malloc(sizeof(double)*nx*ny);
  state->rho = (double*)malloc(sizeof(double)*nx*ny);

#pragma omp parallel for
  for(int ii = 0; ii < ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < nx; ++jj) {
      const int ind = ii*nx + jj;
      state->x[ind] = 0.0;
      state->r[ind] = 0.0;
      state->p[ind] = 0.0;
      state->b[ind] = 0.0;
      state->Ap[ind] = 0.0;
      state->s_x[ind] = 0.0;
      state->s_y[ind] = 0.0;
      state->rho[ind] = 0.0;
    }
  }
}

// Performs the CG solve
void solve(
    const int nx, const int ny, const double dt, const int niters, double* x, 
    double* r, double* p, const double* rho, double* s_x, double* s_y, 
    double* Ap, double* b, const int* celldx, const int* celldy)
{
  initialise_cg(nx, ny, dt, p, rho, s_x, s_y, Ap, celldx, celldy);

#ifdef DEBUG
  printf("\nVector Ax initial: \n");
  print_vec(nx, nx, Ap);
#endif

  // Store initial residual
  double old_rr = 0.0;
  store_residual(nx, ny, b, Ap, r, &old_rr);

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
    double* s_x, double* s_y, double* Ap, const int* celldx, const int* celldy)
{
  // TODO: Calculating the coefficients with edge centered densities...
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      const int ind = ii*nx+jj;
      s_x[ind] = (dt*URANIUM_CONDUCTIVITY)/
        ((0.5*(rho[ind-1]+rho[ind])*URANIUM_HEAT_CAPACITY)*(celldx[jj]*celldx[jj]));
      s_y[ind] = (dt*URANIUM_CONDUCTIVITY)/
        ((0.5*(rho[ind-nx]+rho[ind])*URANIUM_HEAT_CAPACITY)*(celldy[ii]*celldy[ii]));
    }
  }

  // Initialise the guess at solution vector
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      const int ind = ii*nx+jj;
      Ap[ind] = 
        - s_y[ind-nx]*p[ind-nx]
        - s_x[ind-1]*p[ind-1] 
        + (1.0+s_x[ind-1]+s_x[ind]+s_y[ind-nx]+s_y[ind])*p[ind]
        - s_x[ind]*p[ind+1]
        - s_y[ind]*p[ind+nx];
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
      const int ind = ii*nx + jj;
      p[ind] = r[ind] + beta*p[ind];
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
      const int ind = ii*nx + jj;
      Ap[ind] = 
        - s_y[ind-nx]*p[ind-nx]
        - s_x[ind-1]*p[ind-1] 
        + (1.0+s_x[ind-1]+s_x[ind]+s_y[ind-nx]+s_y[ind])*p[ind]
        - s_x[ind]*p[ind+1]
        - s_y[ind]*p[ind+nx];
      pAp += p[ind]*Ap[ind];
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
      const int ind = ii*nx + jj;
      x[ind] += alpha*p[ind];
      r[ind] -= alpha*Ap[ind];
      rr_temp += r[ind]*r[ind];
    }
  }

  *new_rr = rr_temp;
  return rr_temp / old_rr;
}

// Update the residual at the current step
void store_residual(
    int nx, int ny, double* b, double* Ap, double* r, double* old_rr)
{
  double rr_temp = 0.0;

#pragma omp parallel for reduction(+: rr_temp)
  for(int ii = 0; ii < ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < nx; ++jj) {
      const int ind = ii*nx + jj;
      r[ind] = b[ind] - Ap[ind];
      rr_temp += r[ind]*r[ind];
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

// Stores a matrix in the provided filepath
void store_matrix(int m, int n, double* A, const char* filepath)
{
  FILE* fp = fopen(filepath, "wb");

  for(int ii = 0; ii < m; ++ii) {
    for(int jj = 0; jj < n; ++jj) {
      fprintf(fp, "%d %d %.6e\n", ii, jj, A[ii*n + jj]);
    }
  }
}

