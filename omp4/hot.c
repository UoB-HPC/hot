#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hot.h"
#include "../hot_data.h"
#include "../hot_interface.h"
#include "../../omp4/shared.h"
#include "../../profiler.h"
#include "../../comms.h"

// Performs the CG solve, you always want to perform these steps, regardless
// of the context of the problem etc.
void solve_diffusion_2d(
    const int nx, const int ny, Mesh* mesh, const int max_inners, const double dt, 
    const double heat_capacity, const double conductivity,
    double* x, double* r, double* p, double* rho, double* s_x, double* s_y, 
    double* Ap, int* end_niters, double* end_error, double* reduce_array,
    const double* edgedx, const double* edgedy)
{
  // Store initial residual
  double local_old_r2 = initialise_cg(
      nx, ny, mesh->pad, dt, heat_capacity, conductivity, p, r, x, rho, 
      s_x, s_y, edgedx, edgedy);
  double global_old_r2 = reduce_all_sum(local_old_r2);

  handle_boundary_2d(nx, ny, mesh, p, NO_INVERT, PACK);
  handle_boundary_2d(nx, ny, mesh, x, NO_INVERT, PACK);

  // TODO: Can one of the allreduces be removed with kernel fusion?
  int ii = 0;
  for(ii = 0; ii < max_inners; ++ii) {
    const double local_pAp = calculate_pAp(nx, ny, mesh->pad, s_x, s_y, p, Ap);
    const double global_pAp = reduce_all_sum(local_pAp);
    const double alpha = global_old_r2/global_pAp;

    const double local_new_r2 = calculate_new_r2(nx, ny, mesh->pad, alpha, x, p, r, Ap);
    const double global_new_r2 = reduce_all_sum(local_new_r2);
    const double beta = global_new_r2/global_old_r2;
    handle_boundary_2d(nx, ny, mesh, x, NO_INVERT, PACK);

    // Check if the solution has converged
    if(fabs(global_new_r2) < EPS) {
      global_old_r2 = global_new_r2;
      printf("Successfully converged.\n");
      break;
    }

    update_conjugate(nx, ny, mesh->pad, beta, r, p);
    handle_boundary_2d(nx, ny, mesh, p, NO_INVERT, PACK);

    // Store the old squared residual
    global_old_r2 = global_new_r2;
  }

  *end_niters = ii+1;
  *end_error = global_old_r2;
}

// Initialises the CG solver
double initialise_cg(
    const int nx, const int ny, const int pad, const double dt, 
    const double heat_capacity, const double conductivity, double* p, double* r, 
    const double* x, const double* rho, double* s_x, double* s_y, 
    const double* edgedx, const double* edgedy)
{
  START_PROFILING(&compute_profile);

  // https://inldigitallibrary.inl.gov/sti/3952796.pdf
  // Take the average of the coefficients at the cells surrounding 
  // each face
  int nteams = (int)ceil((nx+1)*ny/(double)NTHREADS);
#ifdef CLANG
#pragma omp target teams distribute parallel for collapse(2) \
  thread_limit(NTHREADS) num_teams(nteams)
#else
#pragma omp target teams distribute parallel for 
#endif
  for(int ii = pad; ii < ny-pad; ++ii) {
    for(int jj = pad; jj < (nx+1)-pad; ++jj) {
      s_x[(ii)*(nx+1)+(jj)] = (dt*conductivity*(rho[(ii)*nx+(jj)]+rho[(ii)*nx+(jj-1)]))/
        (2.0*rho[(ii)*nx+(jj)]*rho[(ii)*nx+(jj-1)]*edgedx[jj]*edgedx[jj]*heat_capacity);
    }
  }

  nteams = (int)ceil(nx*(ny+1)/(double)NTHREADS);
#ifdef CLANG
#pragma omp target teams distribute parallel for collapse(2) thread_limit(NTHREADS) num_teams(nteams)
#else
#pragma omp target teams distribute parallel for
#endif
  for(int ii = pad; ii < (ny+1)-pad; ++ii) {
    for(int jj = pad; jj < nx-pad; ++jj) {
      s_y[(ii)*nx+(jj)] = (dt*conductivity*(rho[(ii)*nx+(jj)]+rho[(ii-1)*nx+(jj)]))/
        (2.0*rho[(ii)*nx+(jj)]*rho[(ii-1)*nx+(jj)]*edgedy[ii]*edgedy[ii]*heat_capacity);
    }
  }

  double initial_r2 = 0.0;
  nteams = (int)ceil(nx*ny/(double)NTHREADS);

#ifdef CLANG
#pragma omp target teams distribute parallel for collapse(2) thread_limit(NTHREADS) \
  num_teams(nteams) map(tofrom:initial_r2) reduction(+: initial_r2)
#else
#pragma omp target teams distribute parallel for reduction(+: initial_r2)
#endif
  for(int ii = pad; ii < ny-pad; ++ii) {
    for(int jj = pad; jj < nx-pad; ++jj) {
      r[(ii)*nx+(jj)] = x[(ii)*nx+(jj)] -
        ((s_y[(ii)*nx+(jj)]+s_x[(ii)*(nx+1)+(jj)]+1.0+
          s_x[(ii)*(nx+1)+(jj+1)]+s_y[(ii+1)*nx+(jj)])*x[(ii)*nx+(jj)]
         - s_y[(ii)*nx+(jj)]*x[(ii-1)*nx+(jj)]
         - s_x[(ii)*(nx+1)+(jj)]*x[(ii)*nx+(jj-1)] 
         - s_x[(ii)*(nx+1)+(jj+1)]*x[(ii)*nx+(jj+1)]
         - s_y[(ii+1)*nx+(jj)]*x[(ii+1)*nx+(jj)]);
      p[(ii)*nx+(jj)] = r[(ii)*nx+(jj)];
      initial_r2 += r[(ii)*nx+(jj)]*r[(ii)*nx+(jj)];
    }
  }

  STOP_PROFILING(&compute_profile, "initialise cg");
  return initial_r2;
}

// Calculates a value for alpha
double calculate_pAp(
    const int nx, const int ny, const int pad, const double* s_x, 
    const double* s_y, double* p, double* Ap)
{
  START_PROFILING(&compute_profile);

  // You don't need to use a matrix as the band matrix is fully predictable
  // from the 5pt stencil
  double pAp = 0.0;
  const int nteams = (int)ceil(nx*ny/(double)NTHREADS);
#ifdef CLANG
#pragma omp target teams distribute parallel for collapse(2) thread_limit(NTHREADS) \
  num_teams(nteams) map(tofrom:pAp) reduction(+: pAp)
#else
#pragma omp target teams distribute parallel for reduction(+: pAp)
#endif
  for(int ii = pad; ii < ny-pad; ++ii) {
    for(int jj = pad; jj < nx-pad; ++jj) {
      Ap[(ii)*nx+(jj)] = 
        (s_y[(ii)*nx+(jj)]+s_x[(ii)*(nx+1)+(jj)]+1.0+
         s_x[(ii)*(nx+1)+(jj+1)]+s_y[(ii+1)*nx+(jj)])*p[(ii)*nx+(jj)]
        - s_y[(ii)*nx+(jj)]*p[(ii-1)*nx+(jj)]
        - s_x[(ii)*(nx+1)+(jj)]*p[(ii)*nx+(jj-1)] 
        - s_x[(ii)*(nx+1)+(jj+1)]*p[(ii)*nx+(jj+1)]
        - s_y[(ii+1)*nx+(jj)]*p[(ii+1)*nx+(jj)];
      pAp += p[(ii)*nx+(jj)]*Ap[(ii)*nx+(jj)];
    }
  }

  STOP_PROFILING(&compute_profile, "calculate alpha");
  return pAp;
}

// Updates the current guess using the calculated alpha
double calculate_new_r2(
    const int nx, const int ny, const int pad, double alpha, double* x, 
    double* p, double* r, double* Ap)
{
  START_PROFILING(&compute_profile);

  double new_r2 = 0.0;

  const int nteams = (int)ceil(nx*ny/(double)NTHREADS);
#ifdef CLANG
#pragma omp target teams distribute parallel for collapse(2) thread_limit(NTHREADS) \
  num_teams(nteams) map(tofrom: new_r2) reduction(+: new_r2)
#else
#pragma omp target teams distribute parallel for reduction(+: new_r2)
#endif
  for(int ii = pad; ii < ny-pad; ++ii) {
    for(int jj = pad; jj < nx-pad; ++jj) {
      x[(ii)*nx+(jj)] += alpha*p[(ii)*nx+(jj)];
      r[(ii)*nx+(jj)] -= alpha*Ap[(ii)*nx+(jj)];
      new_r2 += r[(ii)*nx+(jj)]*r[(ii)*nx+(jj)];
    }
  }

  STOP_PROFILING(&compute_profile, "calculate new r2");
  return new_r2;
}

// Updates the conjugate from the calculated beta and residual
void update_conjugate(
    const int nx, const int ny, const int pad, const double beta, 
    const double* r, double* p)
{
  START_PROFILING(&compute_profile);

  const int nteams = (int)ceil(nx*ny/(double)NTHREADS);
#ifdef CLANG
#pragma omp target teams distribute parallel for collapse(2) \
  thread_limit(NTHREADS) num_teams(nteams) 
#else
#pragma omp target teams distribute parallel for 
#endif
  for(int ii = pad; ii < ny-pad; ++ii) {
    for(int jj = pad; jj < nx-pad; ++jj) {
      p[(ii)*nx+(jj)] = r[(ii)*nx+(jj)] + beta*p[(ii)*nx+(jj)];
    }
  }
  STOP_PROFILING(&compute_profile, "update conjugate");
}

