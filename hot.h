#pragma once

#include "../shared.h"
#include "../mesh.h"

// Arbitrary values for calculating conduction coefficient
#define CONDUCTIVITY 1.0//25.0
#define HEAT_CAPACITY 1.0//100.0
#define MAX_INNER_ITERATIONS 10000

// Performs the CG solve
void solve_diffusion(
    const int nx, const int ny, Mesh* mesh, const double dt, double* x, 
    double* r, double* p, double* rho, double* s_x, double* s_y, 
    double* Ap, int* end_niters, double* end_error, const double* edgedx, 
    const double* edgedy);

// Initialises the CG solver
double initialise_cg(
    const int nx, const int ny, const double dt, double* p, double* r,
    const double* x, const double* rho, double* s_x, double* s_y, 
    const double* edgedx, const double* edgedy);

// Updates the conjugate from the calculated beta and residual
void update_conjugate(
    const int nx, const int ny, const double beta, const double* r, double* p);

// Calculates a value for alpha
double calculate_pAp(
    const int nx, const int ny, const double* s_x, const double* s_y,
    double* p, double* Ap);

// Updates the current guess using the calculated alpha
double calculate_new_r2(
    int nx, int ny, double alpha, double* x, double* p, double* r, double* Ap);

// Prints the vector to std out
void print_vec(
    const int nx, const int ny, double* a);

