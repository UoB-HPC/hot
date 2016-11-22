#pragma once

#define MPI

#include "../shared.h"

#define WIDTH 10.0
#define HEIGHT 10.0
#define MAX_DT 0.04
#define NVARS_TO_COMM 4 // rho, e

// Arbitrary values for calculating conduction coefficient
#define CONDUCTIVITY 25
#define HEAT_CAPACITY 100

// Contains all of the data regarding a particular mesh
typedef struct
{
  int local_nx;
  int local_ny;
  int global_nx;
  int global_ny;
  int niters;
  int rank;
  int nranks;
  int width;
  int height;

  int x_off;
  int y_off;

  int* edgedx;
  int* edgedy;

  int* neighbours;

  double dt;

  double* north_buffer_out;
  double* east_buffer_out;
  double* south_buffer_out;
  double* west_buffer_out;
  double* north_buffer_in;
  double* east_buffer_in;
  double* south_buffer_in;
  double* west_buffer_in;

} Mesh;

// Contains all of the state information for the solver
typedef struct
{
  double* Ap;
  double* e;
  double* r;
  double* x;
  double* p;
  double* s_x;
  double* s_y;
  double* rho;

} State;

// Initialises the mesh
void initialise_mesh(Mesh* mesh);

// Initialises the state variables
void initialise_state(const int nx, const int ny, State* state);

// Initialise the communications
void initialise_comms(
    int argc, char** argv, Mesh* mesh);

// Initialises the CG solver
void initialise_cg(
    const int nx, const int ny, const double dt, double* p, const double* rho, 
    double* s_x, double* s_y, double* Ap, const int* celldx, const int* celldy);

// Performs the CG solve
void solve(
    const int nx, const int ny, const double dt, const int niters, double* x, 
    double* r, double* p, const double* rho, double* s_x, double* s_y, 
    double* Ap, double* e, const int* celldx, const int* celldy);

// Updates the conjugate from the calculated beta and residual
void update_conjugate(
    const int nx, const int ny, const double beta, const double* r, double* p);

// Calculates a value for alpha
double calculate_alpha(
    const int nx, const int ny, const double* s_x, const double* s_y,
    double old_rr, double* p, double* Ap);

// Updates the current guess using the calculated alpha
double calculate_beta(
    int nx, int ny, double alpha, double old_rr, 
    double* x, double* p, double* r, double* Ap, double* new_rr);

// Update the residual at the current step
void store_residual(
    int nx, int ny, double* e, double* Ap, double* r, double* old_rr);

// Copies the vector src into dest
void copy_vec(
    const int nx, const int ny, double* src, double* dest);

// Prints the vector to std out
void print_vec(
    const int nx, const int ny, double* a);

// Stores a matrix in the provided filepath
void store_matrix(
    int m, int n, double* A, const char* filepath);

