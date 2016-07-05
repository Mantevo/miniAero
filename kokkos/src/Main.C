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
#include "Main.h"
#include "TimeSolverExplicitRK4.h"
#include "Parallel3DMesh.h"
#include "MeshData.h"
#include "Options.h"
#include "YAML_Doc.h"
#include "YAML_Default.h"

#include <unistd.h>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include <Kokkos_Threads.hpp>


#if WITH_MPI
#include <mpi.h>
#endif

double test_cuda(const Options & simulation_options);
void add_params_to_yaml(YAML_Doc& doc, Options& options);
void add_configuration_to_yaml(YAML_Doc& doc, int numranks, int numthreads);
void add_runtime_to_yaml(YAML_Doc& doc, double setup_time, double execution_time, double total_time);

int main(int argc, char *argv[])
{
time_t startTime=0, endTime=0;



  time(&startTime);
  int num_procs, my_id;
#if WITH_MPI
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
#else
  num_procs=1;
  my_id=0;
#endif

  Options simulation_options;
  simulation_options.read_options_file();
  YAML_Doc doc("MiniAero", "1.0");
  if(my_id==0){
    add_params_to_yaml(doc, simulation_options);
    add_configuration_to_yaml(doc, num_procs, simulation_options.nthreads);
  }

double execution_time = 0.0;
#if HAVE_CUDA
  execution_time = test_cuda(simulation_options);
#else
  execution_time = test_host(simulation_options);
#endif

  time(&endTime);
  double elapsedTime = difftime(endTime,startTime);
  if(my_id==0){
    fprintf(stdout,"\n ... Total elapsed time: %8.2f seconds ...\n",elapsedTime);
  }
  double setup_time = elapsedTime-execution_time;

  if(my_id==0){
    add_runtime_to_yaml(doc,setup_time, execution_time, elapsedTime);
    doc.generateYAML();
  }
  return 0;
}

double test_host(const Options & simulation_options){

  int num_procs, my_id;
#if WITH_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
#else
  num_procs=1;
  my_id=0;
#endif



//  size_t numa_node_count = 1;
  size_t numa_node_thread_count = simulation_options.nthreads;
  Kokkos::Threads::initialize( numa_node_thread_count );

  //Setup mesh on host
  time_t setupStartTime=0, setupEndTime=0;
  time(&setupStartTime);
  int nx = simulation_options.nx, ny = simulation_options.ny, nz = simulation_options.nz;
  double lx = simulation_options.lx, ly = simulation_options.ly, lz = simulation_options.lz;
  double angle = simulation_options.angle;
  int problem_type = simulation_options.problem_type;

  Parallel3DMesh mesh(nx, ny, nz, lx, ly, lz, problem_type, angle);
  struct MeshData<Kokkos::Threads> mesh_data;
  mesh.fillMeshData<Kokkos::Threads>(mesh_data);
  time(&setupEndTime);
  double setupElapsedTime = difftime(setupEndTime,setupStartTime);
  if(my_id==0){
    fprintf(stdout,"\n ... Setup time: %8.2f seconds ...\n", setupElapsedTime);
  }

  //Run on device
  time_t runStartTime=0, runEndTime=0;
  time(&runStartTime);
  TimeSolverExplicitRK4<Kokkos::Threads> * time_solver = new TimeSolverExplicitRK4<Kokkos::Threads>(mesh_data, simulation_options);
  time_solver->Solve();
  delete time_solver;
  time(&runEndTime);
  double runElapsedTime = difftime(runEndTime,runStartTime);
  if(my_id==0){
    fprintf(stdout,"\n ... Device Run time: %8.2f seconds ...\n", runElapsedTime);
  }


  Kokkos::Threads::finalize();
#if WITH_MPI
  MPI_Finalize();
#endif

  return runElapsedTime;

}

void add_params_to_yaml(YAML_Doc& doc, Options & params)
{
  doc.add("Global Run Parameters","");
  doc.get("Global Run Parameters")->add("mesh cells","");
  doc.get("Global Run Parameters")->get("mesh cells")->add("nx",params.nx);
  doc.get("Global Run Parameters")->get("mesh cells")->add("ny",params.ny);
  doc.get("Global Run Parameters")->get("mesh cells")->add("nz",params.nz);
  doc.get("Global Run Parameters")->add("time step", params.dt);
  doc.get("Global Run Parameters")->add("number of time steps", params.ntimesteps);
  std::string second_order = params.second_order_space ? "2nd" : "1st";
  doc.get("Global Run Parameters")->add("spatial discretization order:", second_order);
  std::string viscous = params.viscous ? "Yes" : "No";
  doc.get("Global Run Parameters")->add("viscous:", viscous);
}

void add_configuration_to_yaml(YAML_Doc& doc, int numprocs, int numthreads)
{
  doc.get("Global Run Parameters")->add("number of MPI Ranks:", numprocs);
  doc.get("Global Run Parameters")->add("number of threads/MPI Rank", numthreads);

  doc.add("Platform","");
  doc.get("Platform")->add("hostname", MINIAERO_HOSTNAME);
  doc.get("Platform")->add("processor", MINIAERO_PROCESSOR);

  doc.add("Build","");
  doc.get("Build")->add("CXX",MINIAERO_CXX);
  doc.get("Build")->add("CXXFLAGS",MINIAERO_CXXFLAGS);

  std::string using_mpi("no");
#ifdef WITH_MPI 
  using_mpi = "yes";
#endif
  doc.get("Build")->add("using MPI",using_mpi);
}

void add_runtime_to_yaml(YAML_Doc& doc, double setup_time, double execution_time, double total_time)
{
  doc.add("Setup Time(Seconds)", setup_time);
  doc.add("Execution Time(Seconds)", execution_time);
  doc.add("Total Time(Seconds)", total_time);
}
