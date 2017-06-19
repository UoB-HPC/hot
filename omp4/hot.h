#pragma once

#include "../../shared.h"
#include "../../mesh.h"

// Initialises the CG solver
double initialise_cg(
    const int nx, const int ny, const int pad, const double dt, 
    const double heat_capacity, const double conductivity, double* p, double* r, 
    const double* x, const double* rho, double* s_x, double* s_y, 
    const double* edgedx, const double* edgedy);

// Calculates a value for alpha
double calculate_pAp(
    const int nx, const int ny, const int pad, const double* s_x, 
    const double* s_y, double* p, double* Ap);

// Updates the current guess using the calculated alpha
double calculate_new_r2(
    const int nx, const int ny, const int pad, double alpha, double* x, 
    double* p, double* r, double* Ap);

// Updates the conjugate from the calculated beta and residual
void update_conjugate(
    const int nx, const int ny, const int pad, const double beta, 
    const double* r, double* p);

