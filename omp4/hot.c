#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hot.h"
#include "../hot_interface.h"
#include "../../profiler.h"
#include "../../comms.h"

// Performs the CG solve, you always want to perform these steps, regardless
// of the context of the problem etc.
void solve_diffusion_2d(
    const int nx, const int ny, Mesh* mesh, const int max_inners, const double dt, 
    double* x, double* r, double* p, double* rho, double* s_x, double* s_y, 
    double* Ap, int* end_niters, double* end_error, double* reduce_array,
    const double* edgedx, const double* edgedy)
{
  // Store initial residual
  double local_old_r2 = initialise_cg(
      nx, ny, dt, p, r, x, rho, s_x, s_y, edgedx, edgedy);
  double global_old_r2 = reduce_all_sum(local_old_r2);

  handle_boundary_2d(nx, ny, mesh, p, NO_INVERT, PACK);
  handle_boundary_2d(nx, ny, mesh, x, NO_INVERT, PACK);

  // TODO: Can one of the allreduces be removed with kernel fusion?
  int ii = 0;
  for(ii = 0; ii < max_inners; ++ii) {
    const double local_pAp = calculate_pAp(nx, ny, s_x, s_y, p, Ap);
    const double global_pAp = reduce_all_sum(local_pAp);
    const double alpha = global_old_r2/global_pAp;

    const double local_new_r2 = calculate_new_r2(nx, ny, alpha, x, p, r, Ap);
    const double global_new_r2 = reduce_all_sum(local_new_r2);
    const double beta = global_new_r2/global_old_r2;
    handle_boundary_2d(nx, ny, mesh, x, NO_INVERT, PACK);

    // Check if the solution has converged
    if(fabs(global_new_r2) < EPS) {
      global_old_r2 = global_new_r2;
      break;
    }

    update_conjugate(nx, ny, beta, r, p);
    handle_boundary_2d(nx, ny, mesh, p, NO_INVERT, PACK);

    // Store the old squared residual
    global_old_r2 = global_new_r2;
  }

  *end_niters = ii;
  *end_error = global_old_r2;
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
#pragma omp target teams distribute parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      s_x[(ii*(nx+1)+jj)] = (dt*CONDUCTIVITY*(rho[(ii*nx+jj)]+rho[(ii*nx+jj)-1]))/
        (2.0*rho[(ii*nx+jj)]*rho[(ii*nx+jj)-1]*edgedx[jj]*edgedx[jj]*HEAT_CAPACITY);
    }
  }
#pragma omp target teams distribute parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      s_y[(ii*nx+jj)] = (dt*CONDUCTIVITY*(rho[(ii*nx+jj)]+rho[(ii*nx+jj)-nx]))/
        (2.0*rho[(ii*nx+jj)]*rho[(ii*nx+jj)-nx]*edgedy[ii]*edgedy[ii]*HEAT_CAPACITY);
    }
  }

  double initial_r2 = 0.0;
#pragma omp target teams distribute parallel for reduction(+: initial_r2)
  for(int ii = PAD; ii < ny-PAD; ++ii) {
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      r[(ii*nx+jj)] = x[(ii*nx+jj)] -
        ((s_y[(ii*nx+jj)]+s_x[(ii*(nx+1)+jj)]+1.0+s_x[(ii*(nx+1)+jj)+1]+s_y[(ii*nx+jj)+nx])*x[(ii*nx+jj)]
         - s_y[(ii*nx+jj)]*x[(ii*nx+jj)-nx]
         - s_x[(ii*(nx+1)+jj)]*x[(ii*nx+jj)-1] 
         - s_x[(ii*(nx+1)+jj)+1]*x[(ii*nx+jj)+1]
         - s_y[(ii*nx+jj)+nx]*x[(ii*nx+jj)+nx]);
      p[(ii*nx+jj)] = r[(ii*nx+jj)];
      initial_r2 += r[(ii*nx+jj)]*r[(ii*nx+jj)];
    }
  }

  STOP_PROFILING(&compute_profile, "initialise cg");
  return initial_r2;
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
#pragma omp target teams distribute parallel for reduction(+: pAp)
  for(int ii = PAD; ii < ny-PAD; ++ii) {
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      Ap[(ii*nx+jj)] = 
        (s_y[(ii*nx+jj)]+s_x[(ii*(nx+1)+jj)]+1.0+s_x[(ii*(nx+1)+jj)+1]+s_y[(ii*nx+jj)+nx])*p[(ii*nx+jj)]
        - s_y[(ii*nx+jj)]*p[(ii*nx+jj)-nx]
        - s_x[(ii*(nx+1)+jj)]*p[(ii*nx+jj)-1] 
        - s_x[(ii*(nx+1)+jj)+1]*p[(ii*nx+jj)+1]
        - s_y[(ii*nx+jj)+nx]*p[(ii*nx+jj)+nx];
      pAp += p[(ii*nx+jj)]*Ap[(ii*nx+jj)];
    }
  }

  STOP_PROFILING(&compute_profile, "calculate alpha");
  return pAp;
}

// Updates the current guess using the calculated alpha
double calculate_new_r2(
    int nx, int ny, double alpha, double* x, double* p, double* r, double* Ap)
{
  START_PROFILING(&compute_profile);

  double new_r2 = 0.0;

#pragma omp target teams distribute parallel for reduction(+: new_r2)
  for(int ii = PAD; ii < ny-PAD; ++ii) {
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      x[(ii*nx+jj)] += alpha*p[(ii*nx+jj)];
      r[(ii*nx+jj)] -= alpha*Ap[(ii*nx+jj)];
      new_r2 += r[(ii*nx+jj)]*r[(ii*nx+jj)];
    }
  }

  STOP_PROFILING(&compute_profile, "calculate new r2");
  return new_r2;
}

// Updates the conjugate from the calculated beta and residual
void update_conjugate(
    const int nx, const int ny, const double beta, const double* r, double* p)
{
  START_PROFILING(&compute_profile);
#pragma omp target teams distribute parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      p[(ii*nx+jj)] = r[(ii*nx+jj)] + beta*p[(ii*nx+jj)];
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

