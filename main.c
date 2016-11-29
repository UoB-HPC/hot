#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hot.h"
#include "../shared/profiler.h"

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
  mesh.width = WIDTH;
  mesh.height = HEIGHT;
  mesh.dt = MAX_DT;
  mesh.rank = MASTER;
  mesh.nranks = 1;
  mesh.niters = atoi(argv[3]);

  initialise_comms(argc, argv, &mesh);
  initialise_mesh(&mesh);
  initialise_state(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, &state);

  write_all_ranks_to_visit(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, mesh.x_off, 
      mesh.y_off, mesh.rank, mesh.nranks, state.x, "initial_result", 0, 0.0);

  struct Profile wallclock = {0};

  int tt = 0;
  double elapsed_sim_time = 0.0;
  for(tt = 0; tt < 30; ++tt) {
    START_PROFILING(&wallclock);
    solve(
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt, mesh.niters, state.x, 
        state.r, state.p, state.rho, state.s_x, state.s_y, 
        state.Ap, mesh.edgedx, mesh.edgedy);
    STOP_PROFILING(&wallclock, "wallclock");

    write_all_ranks_to_visit(
        mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, mesh.x_off, 
        mesh.y_off, mesh.rank, mesh.nranks, state.x, "final_result", tt, elapsed_sim_time);

    elapsed_sim_time += mesh.dt;
  }

  double global_wallclock = 0.0;
  if(tt > 0) {
#ifdef MPI
    struct ProfileEntry pe = profiler_get_profile_entry(&wallclock, "wallclock");
    MPI_Reduce(&pe.time, &global_wallclock, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
#endif
  }

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profile);
    printf("Wallclock %.2fs, Elapsed Simulation Time %.4fs\n", global_wallclock, elapsed_sim_time);
  }

  write_all_ranks_to_visit(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, mesh.x_off, 
      mesh.y_off, mesh.rank, mesh.nranks, state.x, "final_result", 0, elapsed_sim_time);
}

// Initialises the mesh
void initialise_mesh(Mesh* mesh) 
{
  mesh->edgedx = (double*)malloc(sizeof(double)*(mesh->local_nx+1));
  mesh->edgedy = (double*)malloc(sizeof(double)*(mesh->local_ny+1));

#pragma omp parallel for
  for(int ii = 0; ii < mesh->local_nx+1; ++ii) {
    mesh->edgedx[ii] = (double)mesh->width/mesh->global_nx;
  }
#pragma omp parallel for
  for(int ii = 0; ii < mesh->local_ny+1; ++ii) {
    mesh->edgedy[ii] = (double)mesh->width/mesh->global_ny;
  }

  mesh->north_buffer_out 
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->east_buffer_out  
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  mesh->south_buffer_out 
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->west_buffer_out  
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  mesh->north_buffer_in  
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->east_buffer_in   
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  mesh->south_buffer_in  
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->west_buffer_in   
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
}

// Initialises the state variables
void initialise_state(
    const int global_nx, const int global_ny, const int local_nx, const int local_ny, 
    const int xoff, const int yoff, State* state) 
{
  state->Ap = (double*)malloc(sizeof(double)*local_nx*local_ny);
  state->r = (double*)malloc(sizeof(double)*local_nx*local_ny);
  state->x = (double*)malloc(sizeof(double)*local_nx*local_ny);
  state->p = (double*)malloc(sizeof(double)*local_nx*local_ny);
  state->s_x = (double*)malloc(sizeof(double)*(local_nx+1)*(local_ny+1));
  state->s_y = (double*)malloc(sizeof(double)*(local_nx+1)*(local_ny+1));
  state->rho = (double*)malloc(sizeof(double)*local_nx*local_ny);

#pragma omp parallel for
  for(int ii = 0; ii < local_ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < local_nx; ++jj) {
      const int index = ii*local_nx+jj;
      state->x[index] = 0.0;
      state->r[index] = 0.0;
      state->p[index] = 0.0;
      state->Ap[index] = 0.0;
      state->rho[index] = 0.0;
    }
  }
#pragma omp parallel for
  for(int ii = 0; ii < (local_ny+1); ++ii) {
#pragma omp simd
    for(int jj = 0; jj < (local_nx+1); ++jj) {
      state->s_x[ii*(local_nx+1)+jj] = 0.0;
      state->s_y[ii*(local_nx+1)+jj] = 0.0;
    }
  }

  // Set the initial state
#pragma omp parallel for
  for(int ii = 0; ii < local_ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < local_nx; ++jj) {
      const int index = ii*local_nx+jj;
      state->rho[index] = 1.0e3;
      state->x[index] = 1.0e-5*state->rho[index];
    }
  }

  // Crooked pipe problem
#pragma omp parallel for
  for(int ii = 0; ii < local_ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < local_nx; ++jj) {
      const int ioff = ii+yoff;
      const int joff = jj+xoff;
      const int index = ii*local_nx+jj;
      // Box problem
      if((ioff >= 7*global_ny/8 || ioff < global_ny/8) ||
          (joff >= 7*global_nx/8 || joff < global_nx/8)) {
        if(joff > 20) {
          state->rho[index] = 0.1;
          state->x[index] = 0.1*state->rho[index];
        }
        else {
          state->rho[index] = 0.1;
          state->x[index] = 1.0e3*state->rho[index];
        }
      }
    }
  }
}


