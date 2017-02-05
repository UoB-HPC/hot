#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hot_interface.h"
#include "hot_data.h"
#include "../profiler.h"
#include "../comms.h"
#include "../shared.h"
#include "../shared_data.h"
#include "../mesh.h"
#include "../params.h"

int main(int argc, char** argv) 
{
  if(argc < 4) {
    TERMINATE("Usage: ./hot.exe <nx> <ny> <niters>\n");
  }

  Mesh mesh = {0};
  mesh.global_nx = atoi(argv[1]);
  mesh.global_ny = atoi(argv[2]);
  mesh.local_nx = atoi(argv[1]) + 2*PAD;
  mesh.local_ny = atoi(argv[2]) + 2*PAD;
  mesh.width = get_double_parameter("width", ARCH_ROOT_PARAMS);
  mesh.height = get_double_parameter("height", ARCH_ROOT_PARAMS);
  mesh.sim_end = get_double_parameter("sim_end", ARCH_ROOT_PARAMS);
  mesh.dt = get_double_parameter("max_dt", ARCH_ROOT_PARAMS);
  mesh.rank = MASTER;
  mesh.nranks = 1;
  mesh.niters = atoi(argv[3]);

  initialise_mpi(argc, argv, &mesh.rank, &mesh.nranks);
  initialise_devices(mesh.rank);
  initialise_comms(&mesh);
  initialise_mesh_2d(&mesh);

  SharedData shared_data = {0};
  initialise_shared_data_2d(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, &shared_data);

  int tt = 0;
  double elapsed_sim_time = 0.0;
  double wallclock = 0.0;
  for(tt = 0; tt < mesh.niters; ++tt) {
    if(mesh.rank == MASTER) {
      printf("step %d\n", tt+1);
    }

    int end_niters = 0;
    double end_error = 0.0;
    solve_diffusion_2d(
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt, shared_data.x, 
        shared_data.r, shared_data.p, shared_data.rho, shared_data.s_x, 
        shared_data.s_y, shared_data.Ap, &end_niters, &end_error, 
        shared_data.reduce_array, mesh.edgedx, mesh.edgedy);

    if(mesh.rank == MASTER) {
      printf("finished on diffusion iteration %d with error %e\n", 
          end_niters, end_error);
    }

    elapsed_sim_time += mesh.dt;
    if(elapsed_sim_time >= mesh.sim_end) {
      if(mesh.rank == MASTER) {
        printf("reached end of simulation time\n");
      }
      break;
    }
  }

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profile);
    printf("wallclock %.4f, elapsed simulation time %.4fs\n", 
        wallclock, elapsed_sim_time);
  }

  write_all_ranks_to_visit(
      mesh.global_nx+2*PAD, mesh.global_ny+2*PAD, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, mesh.rank, mesh.nranks, mesh.neighbours, 
      shared_data.x, "final_result", 0, elapsed_sim_time);

  finalise_shared_data(&shared_data);
  finalise_mesh(&mesh);
  finalise_comms();
}

