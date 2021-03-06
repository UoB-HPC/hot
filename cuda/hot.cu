#include "../../comms.h"
#include "../../cuda/shared.h"
#include "../../profiler.h"
#include "../hot_data.h"
#include "../hot_interface.h"
#include "hot.h"
#include "hot.k"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Performs the CG solve, you always want to perform these steps, regardless
// of the context of the problem etc.
void solve_diffusion_2d(const int nx, const int ny, Mesh* mesh,
                        const int max_inners, const double dt,
                        const double heat_capacity, const double conductivity,
                        double* temperature, double* r, double* p,
                        double* density, double* s_x, double* s_y, double* Ap,
                        int* end_niters, double* end_error,
                        double* reduce_array, const double* edgedx,
                        const double* edgedy) {

  // Store initial residual
  double local_old_r2 = initialise_cg(nx, ny, mesh->pad, dt, heat_capacity,
                                      conductivity, p, r, temperature, density,
                                      s_x, s_y, reduce_array, edgedx, edgedy);

  double global_old_r2 = reduce_all_sum(local_old_r2);

  handle_boundary_2d(nx, ny, mesh, p, NO_INVERT, PACK);
  handle_boundary_2d(nx, ny, mesh, temperature, NO_INVERT, PACK);

  // TODO: Can one of the allreduces be removed with kernel fusion?
  int ii = 0;
  for (ii = 0; ii < max_inners; ++ii) {

    const double local_pAp =
        calculate_pAp(nx, ny, mesh->pad, s_x, s_y, p, Ap, reduce_array);
    const double global_pAp = reduce_all_sum(local_pAp);
    const double alpha = global_old_r2 / global_pAp;

    const double local_new_r2 = calculate_new_r2(
        nx, ny, mesh->pad, alpha, temperature, p, r, Ap, reduce_array);
    const double global_new_r2 = reduce_all_sum(local_new_r2);
    const double beta = global_new_r2 / global_old_r2;
    handle_boundary_2d(nx, ny, mesh, temperature, NO_INVERT, PACK);

    // Check if the solution has converged
    if (fabs(global_new_r2) < EPS) {
      global_old_r2 = global_new_r2;
      break;
    }

    update_conjugate(nx, ny, mesh->pad, beta, r, p);
    handle_boundary_2d(nx, ny, mesh, p, NO_INVERT, PACK);

    // Store the old squared residual
    global_old_r2 = global_new_r2;
  }

  *end_niters = ii;
  *end_error = global_old_r2;
}

// Initialises the CG solver
double initialise_cg(const int nx, const int ny, const int pad, const double dt,
                     const double heat_capacity, const double conductivity,
                     double* p, double* r, const double* temperature,
                     const double* density, double* s_x, double* s_y,
                     double* reduce_array, const double* edgedx,
                     const double* edgedy) {

  int nblocks = ceil((nx + 1) * ny / (double)NTHREADS);
  calc_s_x<<<nblocks, NTHREADS>>>(nx, ny, pad, dt, heat_capacity, conductivity,
                                  s_x, density, edgedx);
  gpu_check(cudaDeviceSynchronize());

  nblocks = ceil(nx * (ny + 1) / (double)NTHREADS);
  calc_s_y<<<nblocks, NTHREADS>>>(nx, ny, pad, dt, heat_capacity, conductivity,
                                  s_y, density, edgedy);
  gpu_check(cudaDeviceSynchronize());

  nblocks = ceil(nx * ny / (double)NTHREADS);
  calc_initial_r2<<<nblocks, NTHREADS>>>(nx, ny, pad, s_x, s_y, temperature, p,
                                         r, reduce_array);
  gpu_check(cudaDeviceSynchronize());

  double initial_r2 = 0.0;
  finish_sum_reduce(nblocks, reduce_array, &initial_r2);
  return initial_r2;
}

// Calculates a value for alpha
double calculate_pAp(const int nx, const int ny, const int pad,
                     const double* s_x, const double* s_y, double* p,
                     double* Ap, double* reduce_array) {

  START_PROFILING(&compute_profile);
  int nblocks = ceil(nx * ny / (double)NTHREADS);
  calc_pAp<<<nblocks, NTHREADS>>>(nx, ny, pad, s_x, s_y, p, Ap, reduce_array);
  gpu_check(cudaDeviceSynchronize());

  double pAp = 0.0;
  finish_sum_reduce(nblocks, reduce_array, &pAp);
  STOP_PROFILING(&compute_profile, "calculate alpha");
  return pAp;
}

// Updates the current guess using the calculated alpha
double calculate_new_r2(const int nx, const int ny, const int pad, double alpha,
                        double* temperature, double* p, double* r, double* Ap,
                        double* reduce_array) {

  START_PROFILING(&compute_profile);

  int nblocks = ceil(nx * ny / (double)NTHREADS);
  calc_new_r2<<<nblocks, NTHREADS>>>(nx, ny, pad, alpha, temperature, p, r, Ap,
                                     reduce_array);
  gpu_check(cudaDeviceSynchronize());

  double new_r2 = 0.0;
  finish_sum_reduce(nblocks, reduce_array, &new_r2);
  STOP_PROFILING(&compute_profile, "calculate new r2");
  return new_r2;
}

// Updates the conjugate from the calculated beta and residual
void update_conjugate(const int nx, const int ny, const int pad,
                      const double beta, const double* r, double* p) {

  START_PROFILING(&compute_profile);

  int nblocks = ceil(nx * ny / (double)NTHREADS);
  update_p<<<nblocks, NTHREADS>>>(nx, ny, pad, beta, r, p);
  gpu_check(cudaDeviceSynchronize());

  STOP_PROFILING(&compute_profile, "update conjugate");
}
