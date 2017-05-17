#pragma once

#include "../mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

// Performs the CG solve
void solve_diffusion_2d(
    const int nx, const int ny, Mesh* mesh, const int max_inners, const double dt, 
    const double heat_capacity, const double conductivity,
    double* x, double* r, double* p, double* rho, double* s_x, double* s_y, 
    double* Ap, int* end_niters, double* end_error, double* reduce_array,
    const double* edgedx, const double* edgedy);

#ifdef __cplusplus
}
#endif

