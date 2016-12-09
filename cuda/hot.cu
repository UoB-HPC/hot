#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hot.h"
#include "../hot_interface.h"
#include "../../cuda/shared.h"
#include "../../profiler.h"
#include "../../comms.h"
#include "kernels.k"

// Performs the CG solve, you always want to perform these steps, regardless
// of the context of the problem etc.
void solve_diffusion(
    const int nx, const int ny, Mesh* mesh, const double dt, double* x, 
    double* r, double* p, double* rho, double* s_x, double* s_y, 
    double* Ap, int* end_niters, double* end_error, double* reduce_array,
    const double* edgedx, const double* edgedy)
{
  // Store initial residual
  double local_old_r2 = initialise_cg(
      nx, ny, dt, p, r, x, rho, s_x, s_y, reduce_array, edgedx, edgedy);

  double global_old_r2 = reduce_all_sum(
      local_old_r2);

  handle_boundary(nx, ny, mesh, p, NO_INVERT, PACK);
  handle_boundary(nx, ny, mesh, x, NO_INVERT, PACK);

  // TODO: Can one of the allreduces be removed with kernel fusion?
  int ii = 0;
  for(ii = 0; ii < MAX_INNER_ITERATIONS; ++ii) {

    const double local_pAp = calculate_pAp(nx, ny, s_x, s_y, p, Ap, reduce_array);
    const double global_pAp = reduce_all_sum(local_pAp);
    const double alpha = global_old_r2/global_pAp;

    const double local_new_r2 = calculate_new_r2(nx, ny, alpha, x, p, r, Ap, reduce_array);
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
    const double* x, const double* rho, double* s_x, double* s_y, double* reduce_array,
    const double* edgedx, const double* edgedy)
{
  int nblocks = ceil((nx+1)*ny/(double)NTHREADS);
  calc_s_x<<<nblocks, NTHREADS>>>(
      nx, ny, dt, s_x, rho, edgedx);
  gpu_check(cudaDeviceSynchronize());

  nblocks = ceil(nx*(ny+1)/(double)NTHREADS);
  calc_s_y<<<nblocks, NTHREADS>>>(
      nx, ny, dt, s_y, rho, edgedy);
  gpu_check(cudaDeviceSynchronize());

  nblocks = ceil(nx*ny/(double)NTHREADS);
  calc_initial_r2<<<nblocks, NTHREADS>>>(
      nx, ny, s_x, s_y, x, p, r, reduce_array);
  gpu_check(cudaDeviceSynchronize());

  double initial_r2 = 0.0;
  finish_sum_reduce(nblocks, reduce_array, &initial_r2);
  return initial_r2;
}

// Calculates a value for alpha
double calculate_pAp(
    const int nx, const int ny, const double* s_x, 
    const double* s_y, double* p, double* Ap, double* reduce_array)
{
  START_PROFILING(&compute_profile);
  int nblocks = ceil(nx*ny/(double)NTHREADS);
  calc_pAp<<<nblocks, NTHREADS>>>(
      nx, ny, s_x, s_y, p, Ap, reduce_array);
  gpu_check(cudaDeviceSynchronize());

  double pAp = 0.0;
  finish_sum_reduce(nblocks, reduce_array, &pAp);
  STOP_PROFILING(&compute_profile, "calculate alpha");
  return pAp;
}

// Updates the current guess using the calculated alpha
double calculate_new_r2(
    int nx, int ny, double alpha, double* x, double* p, double* r, 
    double* Ap, double* reduce_array)
{
  START_PROFILING(&compute_profile);

  int nblocks = ceil(nx*ny/(double)NTHREADS);
  calc_new_r2<<<nblocks, NTHREADS>>>(nx, ny, alpha, x, p, r, Ap, reduce_array);
  gpu_check(cudaDeviceSynchronize());

  double new_r2 = 0.0;
  finish_sum_reduce(nblocks, reduce_array, &new_r2);
  STOP_PROFILING(&compute_profile, "calculate new r2");
  return new_r2;
}

// Updates the conjugate from the calculated beta and residual
void update_conjugate(
    const int nx, const int ny, const double beta, const double* r, double* p)
{
  START_PROFILING(&compute_profile);

  int nblocks = ceil(nx*ny/(double)NTHREADS);
  update_p<<<nblocks, NTHREADS>>>(nx, ny, beta, r, p);
  gpu_check(cudaDeviceSynchronize());

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

