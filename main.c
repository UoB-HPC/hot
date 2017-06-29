#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
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
  if(argc < 2) {
    TERMINATE("Usage: ./hot.exe <parameter_filename>\n");
  }

  Mesh mesh = {0};
  const char* hot_params = argv[1];
  mesh.global_nx = get_int_parameter("nx", hot_params);
  mesh.global_ny = get_int_parameter("ny", hot_params);
  mesh.pad = 2;
  mesh.local_nx = mesh.global_nx + 2*mesh.pad;
  mesh.local_ny = mesh.global_ny + 2*mesh.pad;
  mesh.width = get_double_parameter("width", ARCH_ROOT_PARAMS);
  mesh.height = get_double_parameter("height", ARCH_ROOT_PARAMS);
  mesh.sim_end = get_double_parameter("sim_end", ARCH_ROOT_PARAMS);
  mesh.dt = get_double_parameter("max_dt", ARCH_ROOT_PARAMS);
  mesh.rank = MASTER;
  mesh.nranks = 1;
  mesh.niters = get_int_parameter("iterations", hot_params);
  const int max_inners = get_int_parameter("max_inners", hot_params);
  const int visit_dump = get_int_parameter("visit_dump", hot_params);

  initialise_mpi(argc, argv, &mesh.rank, &mesh.nranks);
  initialise_devices(mesh.rank);
  initialise_comms(&mesh);
  initialise_mesh_2d(&mesh);

  HotData hot_data = {0};
  initialise_hot_data(&hot_data, hot_params);

  SharedData shared_data = {0};
  initialise_shared_data_2d(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, mesh.pad,
      mesh.x_off, mesh.y_off, mesh.width, mesh.height, hot_params, mesh.edgex, 
      mesh.edgey, &shared_data);

  handle_boundary_2d(
      mesh.local_nx, mesh.local_ny, &mesh, shared_data.rho, NO_INVERT, PACK);
  handle_boundary_2d(
      mesh.local_nx, mesh.local_ny, &mesh, shared_data.e, NO_INVERT, PACK);
  handle_boundary_2d(
      mesh.local_nx, mesh.local_ny, &mesh, shared_data.x, NO_INVERT, PACK);

  if(visit_dump) {
    write_all_ranks_to_visit(
        mesh.global_nx+2*mesh.pad, mesh.global_ny+2*mesh.pad, mesh.local_nx, mesh.local_ny, 
        mesh.pad, mesh.x_off, mesh.y_off, mesh.rank, mesh.nranks, mesh.neighbours, 
        shared_data.x, "final_result", 0, 0.0);
  }

  int tt = 0;
  double elapsed_sim_time = 0.0;
  double wallclock = 0.0;

  for(tt = 0; tt < mesh.niters; ++tt) {
    if(mesh.rank == MASTER) {
      printf("step %d\n", tt+1);
    }

    double w0 = omp_get_wtime();

    int end_niters = 0;
    double end_error = 0.0;
    solve_diffusion_2d(
        mesh.local_nx, mesh.local_ny, &mesh, max_inners, mesh.dt, 
        hot_data.heat_capacity, hot_data.conductivity,  shared_data.x, 
        shared_data.r, shared_data.p, shared_data.rho, shared_data.s_x, 
        shared_data.s_y, shared_data.Ap, &end_niters, &end_error, 
        shared_data.reduce_array0, mesh.edgedx, mesh.edgedy);

    wallclock += omp_get_wtime()-w0;

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

    if(visit_dump) {
      write_all_ranks_to_visit(
          mesh.global_nx+2*mesh.pad, mesh.global_ny+2*mesh.pad, mesh.local_nx, mesh.local_ny, 
          mesh.pad, mesh.x_off, mesh.y_off, mesh.rank, mesh.nranks, mesh.neighbours, 
          shared_data.x, "result", tt, elapsed_sim_time);
    }
  }

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profile);
    printf("wallclock %.4fs, elapsed simulation time %.4fs\n", 
        wallclock, elapsed_sim_time);
  }

  if(visit_dump) {
    write_all_ranks_to_visit(
        mesh.global_nx+2*mesh.pad, mesh.global_ny+2*mesh.pad, mesh.local_nx, mesh.local_ny, 
        mesh.pad, mesh.x_off, mesh.y_off, mesh.rank, mesh.nranks, mesh.neighbours, 
        shared_data.x, "final_result", 1, elapsed_sim_time);
  }

  finalise_shared_data(&shared_data);
  finalise_mesh(&mesh);
  finalise_comms();
}

