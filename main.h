#pragma once

#include "../shared.h"

#define WIDTH 10.0
#define HEIGHT 10.0
#define MAX_DT 0.04
#define NVARS_TO_COMM 4 // rho, b

// Arbitrary values for calculating conduction coefficient
#define CONDUCTIVITY 1.0//25.0
#define HEAT_CAPACITY 1.0//100.0

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

  double* edgedx;
  double* edgedy;

  int neighbours[NNEIGHBOURS];

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
  double* b;
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
void initialise_state(
    const int global_nx, const int global_ny, const int local_nx, 
    const int xoff, const int yoff, const int local_ny, State* state);

// Initialise the communications
void initialise_comms(
    int argc, char** argv, Mesh* mesh);

// Initialises the CG solver
double initialise_cg(
    const int nx, const int ny, const double dt, double* p, double* r,
    const double* x, const double* rho, double* s_x, double* s_y, 
    const double* edgedx, const double* edgedy);

// Performs the CG solve
void solve(
    const int nx, const int ny, Mesh* mesh, const double dt, const int niters, double* x, 
    double* r, double* p, const double* rho, double* s_x, double* s_y, 
    double* Ap, const double* edgedx, const double* edgedy);

void handle_boundary(
    const int nx, const int ny, Mesh* mesh, double* arr, const int pack);

// Updates the conjugate from the calculated beta and residual
void update_conjugate(
    const int nx, const int ny, const double beta, const double* r, double* p);

// Calculates a value for alpha
double calculate_pAp(
    const int nx, const int ny, const double* s_x, const double* s_y,
    double* p, double* Ap);

// Updates the current guess using the calculated alpha
double calculate_new_rr(
    int nx, int ny, double alpha, double* x, double* p, double* r, double* Ap);

// Reduces the value over ranks
double all_reduce(double local_val);

// Prints the vector to std out
void print_vec(
    const int nx, const int ny, double* a);

