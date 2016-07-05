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
#ifndef INCLUDE_TIMESOLVER_EXPLICIT_RK4_H_
#define INCLUDE_TIMESOLVER_EXPLICIT_RK4_H_

// C++ system files
#include <cstdio>
#include <vector>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cassert>

// TPL header files
#include <Kokkos_View.hpp>
#if WITH_MPI
#include <mpi.h>
#endif

//Input File Options
#include "Options.h"

//Data structure headers
#include "MeshData.h"
#include "Faces.h"
#include "Cells.h"

// Functor headers
#include "Extrapolate_BC.h"
#include "Tangent_BC.h"
#include "Inflow_BC.h"
#include "NoSlip_BC.h"
#include "Initial_Conditions.h"
#include "Flux.h"
#include "Roe_Flux.h"
#include "Viscous_Flux.h"
#include "CopyGhost.h"
#include "GreenGauss.h"
#include "StencilLimiter.h"

/* TimeSolverData
 * Class that contains options for time marching.
 */
struct TimeSolverData
{
   unsigned time_it;
   unsigned max_its;
   double start_time;
   double time;
   double dt;

   TimeSolverData() :
      time_it(0),
      max_its(1),
      start_time(0.0),
      time(0.0),
      dt(5e-8)
   {

   }

   ~TimeSolverData()
   {

   }
};

/* update
 * functor that updates solution using old solution, residual and scaling of residual.
 */
template <class Device>
struct update{

  typedef Device     device_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;

  double alpha_;
  solution_field_type res_,soln_,solnp1_;

  update(double alpha, solution_field_type res, solution_field_type soln, solution_field_type solnp1) :
  alpha_(alpha),
  res_(res),
  soln_(soln),
  solnp1_(solnp1){}

  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{
    for(int icomp=0; icomp<5; icomp++)
    {
      solnp1_(i, icomp) = soln_(i, icomp)+alpha_*res_(i, icomp);
    }
  }
};

/* copy
 * functor that copies from one solution array to another.
 */

template <class Device>
struct copy{

  typedef Device     device_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;

  solution_field_type soln_src_,soln_dst_;

  copy(solution_field_type soln_src, solution_field_type soln_dst) :
  soln_src_(soln_src),
  soln_dst_(soln_dst){}

  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{
    for(int icomp=0; icomp<5; icomp++)
    {
      soln_dst_(i, icomp) = soln_src_(i, icomp);
    }
  }
};

/*TimeSolverExplicitRK4
 * Class that runs RK4 solver which is the main loop in this code
 * All of the physics kernels are called from the inner RK4 stage loop.
 * Is a basic implementation of the standard RK4 solver.
 */
template <typename Device>
class TimeSolverExplicitRK4
{
public:
   TimeSolverExplicitRK4(struct MeshData<Device> & input_mesh_data, const Options & options);
   ~TimeSolverExplicitRK4();
   void Solve();
   bool CheckStopCriteria();

private:
  struct MeshData<Device> & mesh_data_;
  TimeSolverData ts_data_;
  unsigned stages_;
  double alpha_[4];
  double beta_[4];
  Options options_;
  TimeSolverExplicitRK4();
};

template <typename Device>
TimeSolverExplicitRK4<Device>::TimeSolverExplicitRK4(struct MeshData<Device> & input_mesh_data, const Options & options):
  ts_data_(),
  mesh_data_(input_mesh_data),
  options_(options)
{
  ts_data_.max_its=options_.ntimesteps;
  ts_data_.dt=options_.dt;
  stages_=4;
  alpha_[0] = 0.0;
  alpha_[1] = 1.0/2.0;
  alpha_[2] = 1.0/2.0;
  alpha_[3] = 1.0;
  beta_[0] = 1.0/6.0;
  beta_[1] = 1.0/3.0;
  beta_[2] = 1.0/3.0;
  beta_[3] = 1.0/6.0;
}


template <typename Device>
TimeSolverExplicitRK4<Device>::~TimeSolverExplicitRK4()
{

}

// ////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Device>
void TimeSolverExplicitRK4<Device>::Solve()
{
  //Kokkos Arrays
  typedef typename ViewTypes<Device>::scalar_field_type scalar_field_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::vector_field_type vector_field_type;
  typedef typename ViewTypes<Device>::gradient_field_type gradient_field_type;
  typedef typename Kokkos::View<int *, Device> id_map_type;

   const double midx=options_.lx/2.0;
   double inflow_state[5];
   inflow_state[0]=0.5805;
   inflow_state[1]=503.96;
   inflow_state[2]=0.0;
   inflow_state[3]=0.0;
   inflow_state[4]=343750.0;

   //Faces - Interior and BC
   //Internal Faces
   Faces<Device> internal_faces = mesh_data_.internal_faces;
   const int ninternal_faces= internal_faces.nfaces_;

   //Boundary Faces
   std::vector<Faces<Device> > extrapolate_faces, tangent_faces, inflow_faces, noslip_faces;
   std::vector<Faces<Device> * > bc_faces;


   typename std::vector<std::pair<std::string, Faces<Device> > >::iterator bc_iter, bc_iter_end;
   bc_iter = mesh_data_.boundary_faces.begin();
   bc_iter_end = mesh_data_.boundary_faces.end();
   for(;bc_iter != bc_iter_end; bc_iter++){
     bc_faces.push_back(&(bc_iter->second));
     if (bc_iter->first == "Extrapolate")
       extrapolate_faces.push_back(bc_iter->second);
     else if(bc_iter->first == "Tangent")
       tangent_faces.push_back(bc_iter->second);
     else if(bc_iter->first == "Inflow")
       inflow_faces.push_back(bc_iter->second);
     else if(bc_iter->first == "NoSlip")
       noslip_faces.push_back(bc_iter->second);
   }

   //Cells
   Cells<Device> cells = mesh_data_.mesh_cells;
   const int nowned_cells = mesh_data_.num_owned_cells;
   const int num_ghosts = mesh_data_.num_ghosts;
   const int ncells = cells.ncells_;

   //Solution Variables
   solution_field_type res_vec("residual", ncells);
   solution_field_type sol_n_vec("solution_n", ncells);
   solution_field_type sol_np1_vec("solution_np1", ncells);
   solution_field_type sol_temp_vec("solution_temp", ncells); //Needed for RK4 Stages.
   gradient_field_type gradients("gradients", ncells);
   solution_field_type limiters("limiters", ncells);


   typename solution_field_type::HostMirror solution_vec = Kokkos::create_mirror(sol_n_vec);
   typename solution_field_type::HostMirror residuals_host = Kokkos::create_mirror(res_vec);
   typename vector_field_type::HostMirror coordinates_host = Kokkos::create_mirror(cells.coordinates_);


   //setup ghosting information

   #ifdef WITH_MPI
     int num_procs_, my_id_;
     MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);
     MPI_Comm_rank(MPI_COMM_WORLD, &my_id_);
     std::vector<int> & sendCount = mesh_data_.sendCount;
     std::vector<int> & recvCount = mesh_data_.recvCount;
     int total_send_count=0, total_recv_count=0;
     for(int i=0; i<sendCount.size(); ++i)
       total_send_count+=sendCount[i];
     for(int i=0; i<recvCount.size(); ++i)
       total_recv_count+=recvCount[i];
     assert(num_ghosts==total_recv_count);

     scalar_field_type ghosted_conserved_vars("ghosted_conserved_vars", total_recv_count*5);
     typename scalar_field_type::HostMirror ghosted_conserved_vars_host = Kokkos::create_mirror(ghosted_conserved_vars);
     scalar_field_type shared_conserved_vars("shared_conserved_vars", total_send_count*5);
     typename scalar_field_type::HostMirror shared_conserved_vars_host = Kokkos::create_mirror(shared_conserved_vars);

     id_map_type send_local_ids = mesh_data_.send_local_ids;
     id_map_type recv_local_ids = mesh_data_.recv_local_ids;
   #endif


   //Gradients
   GreenGauss<Device> green_gauss_gradient(internal_faces, bc_faces, cells, mesh_data_, total_send_count, total_recv_count);

   //Stencil Limiter
   StencilLimiter<Device> stencil_limiter(internal_faces, bc_faces, cells, mesh_data_, total_send_count, total_recv_count);

   if(options_.problem_type == 0)
   {
     initialize_sod3d<Device> init_fields(cells, sol_n_vec, sol_temp_vec, midx);
     parallel_for(nowned_cells, init_fields);
   }
   else
   {
     initialize_constant<Device> init_fields(cells, sol_n_vec, sol_temp_vec, &inflow_state[0]);
     parallel_for(nowned_cells, init_fields);
   }
   // Initialize the value for np1 solution which will be updated each RK stage and then copied to n solution at end of timestep.
   copy<Device> copy_solution( sol_n_vec, sol_np1_vec);
   parallel_for(nowned_cells, copy_solution);

   Device::fence();

   for (ts_data_.time_it = 1; ts_data_.time_it <= ts_data_.max_its; ++ts_data_.time_it)
   {
      // Increment the time, do not need to worry about updating it for stages because no source terms depend on the time.
      ts_data_.time += ts_data_.dt;

      // Print time step info
      if(ts_data_.time_it % options_.output_frequency == 0  && my_id_==0){
        fprintf(stdout, "\nTime Step #%i:  Time = %16.9e; dt = %16.9e\n", ts_data_.time_it, ts_data_.time,
             ts_data_.dt);
      }

      // R-K stages loop ****************************************************
      for (unsigned irk = 0; irk < stages_; ++irk)
      {
        //Update temporary solution used to evaluate the residual for this RK stage
        update<Device> update_rk_stage(alpha_[irk], res_vec, sol_n_vec, sol_temp_vec);
        parallel_for(nowned_cells, update_rk_stage);
        Device::fence();

        #ifdef WITH_MPI
        // Update ghosted values (using sol_temp_vec since it is used for all residual calculations.)

          //copy values to be send from device to host
          extract_shared_vector<Device, 5> extract_shared_values(sol_temp_vec, send_local_ids, shared_conserved_vars);
          parallel_for(num_ghosts,extract_shared_values);
          Device::fence();
          Kokkos::deep_copy(shared_conserved_vars_host, shared_conserved_vars);

          communicate_ghosted_cell_data(sendCount, recvCount, shared_conserved_vars_host.ptr_on_device(),ghosted_conserved_vars_host.ptr_on_device(), 5);

          //copy values to be sent from host to device
          Kokkos::deep_copy(ghosted_conserved_vars, ghosted_conserved_vars_host);
          insert_ghost_vector<Device, 5> insert_ghost_values(sol_temp_vec, recv_local_ids, ghosted_conserved_vars);
          parallel_for(num_ghosts, insert_ghost_values);
          Device::fence();
        #endif

        //Zero fluxes
        zero_cell_flux<Device> zero_flux(cells);
        parallel_for(nowned_cells, zero_flux);
        Device::fence();

        //Compute Gradients and Limiters
        if(options_.second_order_space || options_.viscous){
          green_gauss_gradient.compute_gradients(sol_temp_vec, gradients);
#ifdef WITH_MPI
          green_gauss_gradient.communicate_gradients(gradients);
#endif
          stencil_limiter.compute_min_max(sol_temp_vec);
#ifdef WITH_MPI
          stencil_limiter.communicate_min_max();
#endif
          stencil_limiter.compute_limiter(sol_temp_vec, limiters, gradients);
#ifdef WITH_MPI
          stencil_limiter.communicate_limiter(limiters);
#endif
        }


        //Compute internal face fluxes
        roe_flux<Device> inviscid_flux_evaluator;
        if(options_.viscous){
          newtonian_viscous_flux<Device> viscous_flux_evaluator;
          if(options_.second_order_space){
            compute_face_flux<Device, true, roe_flux<Device>, newtonian_viscous_flux<Device> > fluxop(internal_faces, sol_temp_vec, gradients, limiters, cells, inviscid_flux_evaluator, viscous_flux_evaluator);
            parallel_for(ninternal_faces,fluxop);
          }
          else{
            compute_face_flux<Device, false, roe_flux<Device>, newtonian_viscous_flux<Device> > fluxop(internal_faces, sol_temp_vec, gradients, limiters, cells, inviscid_flux_evaluator, viscous_flux_evaluator);
            parallel_for(ninternal_faces,fluxop);
          }
          Device::fence();
        }
        else{
          no_viscous_flux<Device> viscous_flux_evaluator;
          if(options_.second_order_space){
            compute_face_flux<Device, true, roe_flux<Device>, no_viscous_flux<Device> > fluxop(internal_faces, sol_temp_vec, gradients, limiters, cells, inviscid_flux_evaluator, viscous_flux_evaluator);
            parallel_for(ninternal_faces,fluxop);
          }
          else{
            compute_face_flux<Device, false, roe_flux<Device>, no_viscous_flux<Device> > fluxop(internal_faces, sol_temp_vec, gradients, limiters, cells, inviscid_flux_evaluator, viscous_flux_evaluator);
            parallel_for(ninternal_faces,fluxop);
          }
          Device::fence();
        }

        //Extrapolated BC fluxes
        typename std::vector<Faces<Device> >::iterator ef_iter, ef_iter_end;
        ef_iter = extrapolate_faces.begin();
        ef_iter_end = extrapolate_faces.end();
        for(; ef_iter != ef_iter_end; ++ef_iter){
          Faces<Device> bc_faces = *ef_iter;
          const int nboundary_faces = bc_faces.nfaces_;
          compute_extrapolateBC_flux<Device, roe_flux<Device> > boundary_fluxop(bc_faces, sol_temp_vec, cells, inviscid_flux_evaluator);
          parallel_for(nboundary_faces,boundary_fluxop);
        }
        Device::fence();

        //Tangent BC fluxes
        typename std::vector<Faces<Device> >::iterator tf_iter, tf_iter_end;
        tf_iter = tangent_faces.begin();
        tf_iter_end = tangent_faces.end();
        for(; tf_iter != tf_iter_end; ++tf_iter){
          Faces<Device> bc_faces = *tf_iter;
          const int nboundary_faces = bc_faces.nfaces_;
          compute_tangentBC_flux<Device, roe_flux<Device> > boundary_fluxop(bc_faces, sol_temp_vec, cells, inviscid_flux_evaluator);
          parallel_for(nboundary_faces,boundary_fluxop);
        }
        Device::fence();

        //Noslip BC fluxes
        typename std::vector<Faces<Device> >::iterator if_iter, if_iter_end;
        if_iter = noslip_faces.begin();
        if_iter_end = noslip_faces.end();
        for(; if_iter != if_iter_end; ++if_iter){
          newtonian_viscous_flux<Device> viscous_flux_evaluator;
          Faces<Device> bc_faces = *if_iter;
          const int nboundary_faces = bc_faces.nfaces_;
          compute_NoSlipBC_flux<Device, roe_flux<Device>, newtonian_viscous_flux<Device> > boundary_fluxop(bc_faces, sol_temp_vec, cells, inviscid_flux_evaluator, viscous_flux_evaluator);
          parallel_for(nboundary_faces,boundary_fluxop);
        }
        Device::fence();

        //Inflow BC fluxes
        typename std::vector<Faces<Device> >::iterator nsf_iter, nsf_iter_end;
        nsf_iter = inflow_faces.begin();
        nsf_iter_end = inflow_faces.end();
        for(; nsf_iter != nsf_iter_end; ++nsf_iter){
          Faces<Device> bc_faces = *nsf_iter;
          const int nboundary_faces = bc_faces.nfaces_;
          compute_inflowBC_flux<Device, roe_flux<Device> > boundary_fluxop(bc_faces, sol_temp_vec, cells, &inflow_state[0], inviscid_flux_evaluator);
          parallel_for(nboundary_faces,boundary_fluxop);
        }
        Device::fence();

        //Sum up all of the contributions
        apply_cell_flux<Device> flux_residual(cells, res_vec, ts_data_.dt);
        parallel_for(nowned_cells, flux_residual);
        Device::fence();
 
        //Update np1 solution with each stages contribution
        update<Device> update_fields(beta_[irk],res_vec,sol_np1_vec,sol_np1_vec);
        parallel_for(nowned_cells, update_fields);
        Device::fence();
      }
      // Update the solution vector after having run all of the RK stages.
      copy<Device> copy_solution( sol_np1_vec, sol_n_vec);
      parallel_for(nowned_cells, copy_solution);

   }

   //Output to file on the host.  Requires a deep copy from device to host.
   if(options_.output_results){
     //Copy to host
     Kokkos::deep_copy(solution_vec, sol_n_vec);
     Kokkos::deep_copy(residuals_host, res_vec);
     Kokkos::deep_copy(coordinates_host, cells.coordinates_);

     std::ofstream output_file;
     std::stringstream fs;
     fs << "results.";
     fs << my_id_;
     std::string filename = fs.str();
     output_file.open(filename.c_str(), std::ios::out);

     for(int i=0; i<nowned_cells; i++)
     {
       output_file << coordinates_host(i,0) << "\t";
       output_file << coordinates_host(i,1) << "\t";
       output_file << coordinates_host(i,2) << "\t";
       for(int icomp=0; icomp<5; icomp++){
         output_file << solution_vec(i,icomp) << "\t";
       }
       output_file << std::endl;
     }
     output_file.close();

     std::ofstream gradient_file;
     std::stringstream fs2;
     fs2 << "gradients.";
     fs2 << my_id_;
     std::string filename2 = fs2.str();
     gradient_file.open(filename2.c_str(), std::ios::out);

     for(int i=0; i<nowned_cells; i++)
     {
       gradient_file << coordinates_host(i,0) << "\t";
       gradient_file << coordinates_host(i,1) << "\t";
       gradient_file << coordinates_host(i,2) << "\t";
       for(int icomp=0; icomp<5; icomp++){
         gradient_file << gradients(i,icomp,1) << "\t";
       }
       gradient_file << std::endl;
     }

     std::ofstream limiter_file;
     std::stringstream fs3;
     fs3 << "limiters.";
     fs3 << my_id_;
     std::string filename3 = fs3.str();
     limiter_file.open(filename3.c_str(), std::ios::out);

     for(int i=0; i<nowned_cells; i++)
     {
       limiter_file << coordinates_host(i,0) << "\t";
       limiter_file << coordinates_host(i,1) << "\t";
       limiter_file << coordinates_host(i,2) << "\t";
       for(int icomp=0; icomp<5; icomp++){
         limiter_file << limiters(i,icomp) << "\t";
       }
       limiter_file << std::endl;
     }
   }
}

#endif
