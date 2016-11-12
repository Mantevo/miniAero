/*Copyright (2014) Sandia Corporation.
*Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
*the U.S. Government retains certain rights in this software.
*
*Redistribution and use in source and binary forms, with or without modification,
* are permitted provided that the following conditions are met:
*
*1. Redistributions of source code must retain the above copyright notice,
*this list of conditions and the following disclaimer.
*
*2. Redistributions in binary form must reproduce the above copyright notice,
*this list of conditions and the following disclaimer in the documentation
*and/or other materials provided with the distribution.
*
*3. Neither the name of the copyright holder nor the names of its contributors
*may be used to endorse or promote products derived from this software
*without specific prior written permission.
*
*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
*ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
*IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
*DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
*LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
*SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*/
#if WITH_MPI
#include <mpi.h>
#endif
#include "Main.h"
#include "TimeSolverExplicitRK4.h"
#include "Parallel3DMesh.h"
#include "MeshData.h"
#include "Options.h"

#include <unistd.h>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>

#ifdef MINIAERO_FPMATH_CHECK
#define _GNU_SOURCE
#include <fenv.h>
#include <xmmintrin.h>
#endif

#include <Kokkos_Core.hpp>

int main(int argc, char *argv[])
{
#ifdef MINIAERO_FPMATH_CHECK
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif
    
  int num_procs, my_id;
#if WITH_MPI
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  double startTime = 0.0, endTime = 0.0;
  startTime = MPI_Wtime();
#else
  time_t startTime=0, endTime=0;
  time(&startTime);
  num_procs=1;
  my_id=0;
#endif

  Options simulation_options;
  simulation_options.read_options_file();

  Kokkos::InitArguments init_args;

  char* env_omp_threads = getenv("OMP_NUM_THREADS");
  if(NULL == env_omp_threads) {
    init_args.num_threads = simulation_options.nthreads;
  } else {
    const int env_omp_threads_int = atoi(env_omp_threads);

    if(env_omp_threads_int != simulation_options.nthreads) {
        printf("Detected OMP_NUM_THREADS in environment: %d overriding input deck %d\n",
            env_omp_threads_int, simulation_options.nthreads);
    }

    init_args.num_threads = env_omp_threads_int;
  }

  Kokkos::initialize(init_args );
  run_host(simulation_options);

#if WITH_MPI
  endTime = MPI_Wtime();
  double elapsedTime = endTime-startTime;
#else
  time(&endTime);
  double elapsedTime = difftime(endTime,startTime);
#endif
  Kokkos::finalize();
#if WITH_MPI
  MPI_Finalize();
#endif
  if(my_id==0){
    fprintf(stdout,"\n ... Total elapsed time: %8.2f seconds ...\n",elapsedTime);
  }
  return 0;
}

void run_host(const Options & simulation_options){
  int num_procs, my_id;
#if WITH_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  double setupStartTime = 0.0, setupEndTime = 0.0;
  setupStartTime = MPI_Wtime();
#else
  num_procs=1;
  my_id=0;
  time_t setupStartTime=0, setupEndTime=0;
  time(&setupStartTime);
#endif

//  size_t numa_node_count = 1;
  size_t numa_node_thread_count = simulation_options.nthreads;

  //Setup mesh on host
  int nx = simulation_options.nx, ny = simulation_options.ny, nz = simulation_options.nz;
  double lx = simulation_options.lx, ly = simulation_options.ly, lz = simulation_options.lz;
  double angle = simulation_options.angle;
  int problem_type = simulation_options.problem_type;

  Parallel3DMesh mesh(nx, ny, nz, lx, ly, lz, problem_type, angle);
  struct MeshData<Kokkos::DefaultExecutionSpace> mesh_data;
  mesh.fillMeshData<Kokkos::DefaultExecutionSpace>(mesh_data);
  double setupElapsedTime = 0.0;
#if WITH_MPI
  setupEndTime = MPI_Wtime();
  setupElapsedTime = setupEndTime-setupStartTime; 
#else
  time(&setupEndTime);
  setupElapsedTime = difftime(setupEndTime,setupStartTime);
#endif
  if(my_id==0){
    fprintf(stdout,"\n ... Setup time: %8.2f seconds ...\n", setupElapsedTime);
  }

  //Run on device
#if WITH_MPI
  double runStartTime=0, runEndTime=0;
  runStartTime = MPI_Wtime();
#else
  time_t runStartTime=0, runEndTime=0;
  time(&runStartTime);
#endif
  TimeSolverExplicitRK4<Kokkos::DefaultExecutionSpace> * time_solver = new TimeSolverExplicitRK4<Kokkos::DefaultExecutionSpace>(mesh_data, simulation_options);
  time_solver->Solve();
  delete time_solver;
 
  double runElapsedTime=0;
#if WITH_MPI
  runEndTime = MPI_Wtime();
  runElapsedTime = runEndTime-runStartTime;
#else
  time(&runEndTime);
  runElapsedTime = difftime(runEndTime,runStartTime);
#endif
}

