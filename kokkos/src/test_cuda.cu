#include <Kokkos_Cuda.hpp>
#include "TimeSolverExplicitRK4.h"
#include "Parallel3DMesh.h"
#include "MeshData.h"
#include "Options.h"

#include <ctime>

void test_cuda(const Options & simulation_options){
  int num_procs, my_id;
#if WITH_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
#else
  num_procs=1;
  my_id=0;
#endif

  Kokkos::Cuda::host_mirror_device_type::initialize();
  Kokkos::Cuda::SelectDevice select_device(0);
  Kokkos::Cuda::initialize( select_device );

  //Setup mesh
  time_t setupStartTime=0, setupEndTime=0;
  time(&setupStartTime);
  int nx = simulation_options.nx, ny = simulation_options.ny, nz = simulation_options.nz;
  double lx = simulation_options.lx, ly = simulation_options.ly, lz = simulation_options.lz;
  double angle = simulation_options.angle;
  int problem_type = simulation_options.problem_type;
  Parallel3DMesh mesh(nx, ny, nz, lx, ly, lz, problem_type, angle);
  struct MeshData<Kokkos::Cuda> mesh_data;
  mesh.fillMeshData<Kokkos::Cuda>(mesh_data);
  time(&setupEndTime);
  double setupElapsedTime = difftime(setupEndTime,setupStartTime);
  if(my_id==0){
    fprintf(stdout,"\n ... Setup time: %8.2f seconds ...\n", setupElapsedTime);
  }
  
  
  //Run on device.
  time_t runStartTime=0, runEndTime=0;
  time(&runStartTime);
  TimeSolverExplicitRK4<Kokkos::Cuda> * time_solver = new TimeSolverExplicitRK4<Kokkos::Cuda>(mesh_data, simulation_options);
  time_solver->Solve();
  delete time_solver;
  time(&runEndTime);
  double runElapsedTime = difftime(runEndTime,runStartTime);
  if(my_id==0){
    fprintf(stdout,"\n ... Device Run time: %8.2f seconds ...\n", runElapsedTime);
  }

  Kokkos::Cuda::finalize();
#if WITH_MPI
  MPI_Finalize();
#endif
  return runElapsedTime;
}
