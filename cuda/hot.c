#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hot.h"
#include "../cuda/config.h"
#include "../hot_interface.h"
#include "../../profiler.h"
#include "../../comms.h"

// Performs the CG solve, you always want to perform these steps, regardless
// of the context of the problem etc.
void solve_diffusion(
    const int nx, const int ny, Mesh* mesh, const double dt, double* x, 
    double* r, double* p, double* rho, double* s_x, double* s_y, 
    double* Ap, int* end_niters, double* end_error, const double* edgedx, 
    const double* edgedy)
{
  // Store initial residual
  double local_old_r2 = initialise_cg(
      nx, ny, dt, p, r, x, rho, s_x, s_y, edgedx, edgedy);

  double global_old_r2 = reduce_all_sum(local_old_r2);

  handle_boundary(nx, ny, mesh, p, NO_INVERT, PACK);
  handle_boundary(nx, ny, mesh, x, NO_INVERT, PACK);

  // TODO: Can one of the allreduces be removed with kernel fusion?
  int ii = 0;
  for(ii = 0; ii < MAX_INNER_ITERATIONS; ++ii) {

    const double local_pAp = calculate_pAp(nx, ny, s_x, s_y, p, Ap);
    const double global_pAp = reduce_all_sum(local_pAp);
    const double alpha = global_old_r2/global_pAp;

    const double local_new_r2 = calculate_new_r2(nx, ny, alpha, x, p, r, Ap);
    const double global_new_r2 = reduce_all_sum(local_new_r2);
    const double beta = global_new_r2/global_old_r2;
    handle_boundary(nx, ny, mesh, x, NO_INVERT, PACK);

    // Check if the solution has converged
    if(fabs(global_new_r2) < 1.0e-10) {
      global_old_r2 = global_new_r2;
      break;
    }

    update_conjugate(nx, ny, beta, r, p);
    handle_boundary(nx, ny, mesh, p, NO_INVERT, PACK);

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
  int nthreads_per_block = ceil((nx+1)*ny/(double)NBLOCKS);
  calc_s_x<<<nthreads_per_block, NBLOCKS>>>(
      dt, s_x, rho, edgedx);
  gpu_check(cudaDeviceSynchronize());

  nthreads_per_block = ceil(nx*(ny+1)/(double)NBLOCKS);
  calc_s_y<<<nthreads_per_block, NBLOCKS>>>(
      dt, s_y, rho, edgedy);
  gpu_check(cudaDeviceSynchronize());

  double initial_r2 = 0.0;
  nthreads_per_block = ceil(nx*ny/(double)NBLOCKS);
  calc_initial_r2<<<nthreads_per_block, NBLOCKS>>>(
      dt, s_y, rho, edgedy); 
  gpu_check(cudaDeviceSynchronize());
}

// Calculates a value for alpha
double calculate_pAp(
    const int nx, const int ny, const double* s_x, const double* s_y,
    double* p, double* Ap)
{
  START_PROFILING(&compute_profile);
  double pAp = 0.0;
  calc_pAp<<<nthreads_per_block, NBLOCKS>>>(


  STOP_PROFILING(&compute_profile, "calculate alpha");
  return pAp;
}

// Updates the current guess using the calculated alpha
double calculate_new_r2(
    int nx, int ny, double alpha, double* x, double* p, double* r, double* Ap)
{
  START_PROFILING(&compute_profile);

  double new_r2 = 0.0;

#pragma omp parallel for reduction(+: new_r2)
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      x[ind0] += alpha*p[ind0];
      r[ind0] -= alpha*Ap[ind0];
      new_r2 += r[ind0]*r[ind0];
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

