#include <stdlib.h>
#include <stdio.h>
#include <string.h>
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
  mesh.rank = MASTER;
  mesh.nranks = 1;
  mesh.niters = atoi(argv[3]);

  initialise_mesh(&mesh);
  initialise_state(mesh.local_nx, mesh.local_ny, &state);

  // Fetch the user's parameters
  printf("Using the CG solver\n");

}

// Initialises the mesh
static inline void initialise_mesh(Mesh* mesh) {

}

// Initialises the state variables
static inline void initialise_state(const int nx, const int ny, State* state) {
  state->Ap = (double*)malloc(sizeof(double)*nx*ny);
  state->b = (double*)malloc(sizeof(double)*nx*ny);
  state->r = (double*)malloc(sizeof(double)*nx*ny);
  state->x = (double*)malloc(sizeof(double)*nx*ny);
  state->p = (double*)malloc(sizeof(double)*nx*ny);

#pragma omp parallel for
  for(int ii = 0; ii < ny; ++ii) {
    for(int jj = 0; jj < nx; ++jj) {
      const int index = ii*nx + jj;
      state->x[index] = 0.0;
      state->r[index] = 0.0;
      state->p[index] = 0.0;
      state->b[index] = 0.0;
      state->Ap[index] = 0.0;
    }
  }
}

// Performs the CG solve for the initialise problem
void solve(const int nx, const int ny, double* x, double* r, double* A, double* b, double* p)
{
  // Initialise the guess at solution vector
  for(int ii = PAD; ii < ny-PAD; ++ii) {
  for(int jj = PAD; jj < nx-PAD; ++jj) {
    //Ap[ii] = // A by x
    const int index = ii*nx+jj;
    Ap[index] = A[index]
  }
  }

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

  for(int ii = 0; ii < max_iters; ++ii)
  {
    double alpha = calculate_alpha(
        nx, ny, halo, A.get_mat(), old_rr, p, Ap); 

#ifdef DEBUG
    printf("alpha: %.12e\n", alpha);
#endif

    double new_rr = 0.0;
    double beta = calculate_beta(
        nx, ny, halo, alpha, old_rr, 
        x, p, r, Ap, &new_rr);

#ifdef DEBUG
    printf("beta: %.12e\n", beta);
#endif

    // Check if the solution has converged
    if(fabs(new_rr) < 1.0e-05) {
      printf("exiting at iteration %d with new_rr: %.12e\n", ii, new_rr);
      break;
    }

    update_conjugate(nx, ny, halo, beta, p, r);

    // Store the old squared residual
    old_rr = new_rr;

#ifdef DEBUG
    printf("old_rr: %.12e\n", old_rr);
#endif
  }

  printf("\nResult: \n");
  print_vec(nx, ny, x);
}

// Updates the conjugate from the calculated beta and residual
void update_conjugate(
    int nx, int ny, int halo, double beta, 
    double* p, double* r)
{
#pragma omp parallel for
  for(int ii = halo; ii < ny - halo; ++ii) {
    for(int jj = halo; jj < nx - halo; ++jj) {
      const int index = ii*nx + jj;
      p[index] = r[index] + beta*p[index];
    }
  }
}

// Calculates a value for alpha
double calculate_alpha(
    int nx, int ny, int halo, double* A, 
    double old_rr, double* p, double* Ap)
{
  double pAp = 0.0;

#pragma omp parallel for reduction(+: pAp)
  for(int ii = halo; ii < ny - halo; ++ii) {
    for(int jj = halo; jj < nx - halo; ++jj) {
      const int index = ii*nx + jj;
      double Ap_current = smvp_row(index, A, p);
      Ap[index] = Ap_current;
      pAp += p[index]*Ap_current;
    }
  }

  return old_rr / pAp;
}

// Updates the current guess using the calculated alpha
double calculate_beta(
    int nx, int ny, int halo, double alpha, double old_rr, 
    double* x, double* p, double* r, double* Ap, double* new_rr)
{
  double rr_temp = 0.0;

#pragma omp parallel for reduction(+: rr_temp)
  for(int ii = halo; ii < ny - halo; ++ii) {
    for(int jj = halo; jj < nx - halo; ++jj) {
      const int index = ii*nx + jj;
      x[index] += alpha*p[index];
      r[index] -= alpha*Ap[index];
      rr_temp += r[index]*r[index];
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
    for(int jj = 0; jj < nx; ++jj) {
      const int index = ii*nx + jj;
      r[index] = b[index] - Ap[index];
      rr_temp += r[index]*r[index];
    }
  }

  *old_rr = rr_temp;
}

// Copies the vector src into dest
void copy_vec(
    const int nx, const int ny, double* src, double* dest)
{
#pragma omp parallel for
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

