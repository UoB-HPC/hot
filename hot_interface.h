#pragma once

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

