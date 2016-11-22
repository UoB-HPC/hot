#pragma once

#define MASTER 0
#define PAD 2
#define URANIUM_CONDUCTIVITY 27.5
#define URANIUM_HEAT_CAPACITY 117.2304

#define strmatch(a, b) (strcmp(a, b) == 0)
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

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

  int* celldx;
  int* celldy;

  double dt;

} Mesh;

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
void initialise_state(const int nx, const int ny, State* state);

// Initialises the CG solver
void initialise_cg(
    const int nx, const int ny, const double dt, double* p, const double* rho, 
    double* s_x, double* s_y, double* Ap, const int* celldx, const int* celldy);

// Performs the CG solve
void solve(
    const int nx, const int ny, const double dt, const int niters, double* x, 
    double* r, double* p, const double* rho, double* s_x, double* s_y, 
    double* Ap, double* b, const int* celldx, const int* celldy);

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
    int nx, int ny, double* b, double* Ap, double* r, double* old_rr);

// Copies the vector src into dest
void copy_vec(
    const int nx, const int ny, double* src, double* dest);

// Prints the vector to std out
void print_vec(
    const int nx, const int ny, double* a);

// Stores a matrix in the provided filepath
void store_matrix(
    int m, int n, double* A, const char* filepath);

