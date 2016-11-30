#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hot.h"
#include "../profiler.h"
#include "../comms.h"

#define ind0 (ii*nx + jj)
#define ind1 (ii*(nx+1) + jj)

// Performs the CG solve, you always want to perform these steps, regardless
// of the context of the problem etc.
void solve_diffusion(
    const int nx, const int ny, Mesh* mesh, const double dt, double* x, 
    double* r, double* p, const double* rho, double* s_x, double* s_y, 
    double* Ap, int* end_niters, double* end_error, const double* edgedx, 
    const double* edgedy)
{
  // Store initial residual
  double local_old_rr = initialise_cg(
      nx, ny, dt, p, r, x, rho, s_x, s_y, edgedx, edgedy);

  double global_old_rr = reduce_all_sum(local_old_rr);

  handle_boundary(nx, ny, mesh, p, NO_INVERT, PACK);
  handle_boundary(nx, ny, mesh, x, NO_INVERT, PACK);

  // TODO: Can one of the allreduces be removed if you use the local_rr more?
  int ii = 0;
  for(ii = 0; ii < MAX_INNER_ITERATIONS; ++ii) {
    const double local_pAp = calculate_pAp(nx, ny, s_x, s_y, p, Ap);

    double global_pAp = reduce_all_sum(local_pAp);

    const double alpha = global_old_rr/global_pAp;
    const double local_new_rr = calculate_new_rr(nx, ny, alpha, x, p, r, Ap);

    double global_new_rr = reduce_all_sum(local_new_rr);

    // Check if the solution has converged
    if(fabs(global_new_rr) < 1.0e-05) {
      global_old_rr = global_new_rr;
      break;
    }

    const double beta = global_new_rr/global_old_rr;
    update_conjugate(nx, ny, beta, r, p);

    handle_boundary(nx, ny, mesh, p, NO_INVERT, PACK);
    handle_boundary(nx, ny, mesh, x, NO_INVERT, PACK);

    // Store the old squared residual
    global_old_rr = global_new_rr;
  }

  *end_niters = ii;
  *end_error = global_old_rr;
}

// Initialises the CG solver
double initialise_cg(
    const int nx, const int ny, const double dt, double* p, double* r,
    const double* x, const double* rho, double* s_x, double* s_y, 
    const double* edgedx, const double* edgedy)
{
  START_PROFILING(&compute_profile);

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

  double initial_rr = 0.0;
#pragma omp parallel for reduction(+: initial_rr)
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      r[ind0] = x[ind0] -
        ((1.0+s_y[ind1]+s_x[ind1]+s_x[ind1+1]+s_y[ind1+(nx+1)])*x[ind0]
         - s_y[ind1]*x[ind0-nx]
         - s_x[ind1]*x[ind0-1] 
         - s_x[ind1+1]*x[ind0+1]
         - s_y[ind1+(nx+1)]*x[ind0+nx]);
      p[ind0] = r[ind0];
      initial_rr += r[ind0]*r[ind0];
    }
  }

  STOP_PROFILING(&compute_profile, "initialise cg");
  return initial_rr;
}

// Calculates a value for alpha
double calculate_pAp(
    const int nx, const int ny, const double* s_x, const double* s_y,
    double* p, double* Ap)
{
  START_PROFILING(&compute_profile);

  // You don't need to use a matrix as the band matrix is fully predictable
  // from the 5pt stencil
  double pAp = 0.0;
#pragma omp parallel for reduction(+: pAp)
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      Ap[ind0] = 
        (1.0+s_y[ind1]+s_x[ind1]+s_x[ind1+1]+s_y[ind1+(nx+1)])*p[ind0]
        - s_y[ind1]*p[ind0-nx]
        - s_x[ind1]*p[ind0-1] 
        - s_x[ind1+1]*p[ind0+1]
        - s_y[ind1+(nx+1)]*p[ind0+nx];
      pAp += p[ind0]*Ap[ind0];
    }
  }

  STOP_PROFILING(&compute_profile, "calculate alpha");
  return pAp;
}

// Updates the current guess using the calculated alpha
double calculate_new_rr(
    int nx, int ny, double alpha, double* x, double* p, double* r, double* Ap)
{
  START_PROFILING(&compute_profile);

  double new_rr = 0.0;

#pragma omp parallel for reduction(+: new_rr)
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      x[ind0] += alpha*p[ind0];
      r[ind0] -= alpha*Ap[ind0];
      new_rr += r[ind0]*r[ind0];
    }
  }

  STOP_PROFILING(&compute_profile, "calculate new rr");
  return new_rr;
}

// Updates the conjugate from the calculated beta and residual
void update_conjugate(
    const int nx, const int ny, const double beta, const double* r, double* p)
{
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      p[ind0] = r[ind0] + beta*p[ind0];
    }
  }
  STOP_PROFILING(&compute_profile, "update conjugate");
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
