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
/*
 * GreenGauss.h
 *
 *  Created on: Apr 7, 2014
 *      Author: kjfrank
 */
#include <cstdio>
#include <Kokkos_View.hpp>
#include "Faces.h"

#ifdef ATOMICS_FLUX
#include "Kokkos_Atomic.hpp"
#endif

#ifndef GREENGAUSS_H_
#define GREENGAUSS_H_


/*green_gauss_face
 * functor to compute internal face contributions for Green-Gauss gradient computation.
 */

template <class Device>
struct green_gauss_face{
  typedef Device device_type;
  typedef typename ViewTypes<Device>::scalar_field_type scalar_field_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::face_cell_conn_type face_cell_conn_type;
  typedef typename ViewTypes<Device>::vector_field_type vector_field_type;
  typedef typename ViewTypes<Device>::gradient_storage_field_type gradient_storage_field_type;

  scalar_field_type cell_volumes_;
  face_cell_conn_type face_cell_conn_;
  face_cell_conn_type cell_flux_index_;
  solution_field_type cell_values_;
  gradient_storage_field_type cell_gradient_;
  vector_field_type face_normal_;

  green_gauss_face(Faces<Device> faces, solution_field_type cell_values, Cells<Device> cells):
    cell_volumes_(cells.volumes_),
    face_cell_conn_(faces.face_cell_conn_),
    cell_flux_index_(faces.cell_flux_index_),
    cell_values_(cell_values),
    cell_gradient_(cells.cell_gradient_),
    face_normal_(faces.face_normal_)
  {}

KOKKOS_INLINE_FUNCTION
void operator()( int i )const{
  int left_index = face_cell_conn_(i,0);
  int right_index = face_cell_conn_(i,1);

  double gradient[5][3];
  double conservative_l[5], conservative_r[5];
  double primitives_l[5], primitives_r[5];

  for(int icomp = 0; icomp < 5; ++icomp)
  {
    conservative_l[icomp] = cell_values_(left_index, icomp);
    conservative_r[icomp] = cell_values_(right_index, icomp);
  }
  ComputePrimitives<device_type>(conservative_l, primitives_l);
  ComputePrimitives<device_type>(conservative_r, primitives_r);

  for(int icomp = 0; icomp < 5; ++icomp)
  {
    for(int idir = 0; idir < 3; ++idir)
    {
    gradient[icomp][idir] = 0.5*(primitives_l[icomp]+primitives_r[icomp])*face_normal_(i,idir);
    }
  }

#ifdef ATOMICS_FLUX
  for (int icomp = 0; icomp < 5; ++icomp)
  {
    for(int idir = 0; idir < 3; ++idir)
    {
      double * left_cell = &cell_gradient_(left_index,0,icomp,idir);
      Kokkos::atomic_add(left_cell, gradient[icomp][idir]/cell_volumes_(left_index));
      double * right_cell = &cell_gradient_(right_index,0,icomp,idir);
      Kokkos::atomic_add(right_cell, -gradient[icomp][idir]/cell_volumes_(right_index));
    }
  }
#endif

#ifdef CELL_FLUX
for (int icomp = 0; icomp < 5; ++icomp)
{
  for(int idir = 0; idir < 3; ++idir)
  {
    cell_gradient_(left_index,cell_flux_index_(i,0),icomp, idir) = gradient[icomp][idir]/cell_volumes_(left_index);
    cell_gradient_(right_index,cell_flux_index_(i,1),icomp, idir) = -gradient[icomp][idir]/cell_volumes_(right_index);
  }
}
#endif
}

};

/*green_gauss_boundary_face
 * functor to compute boundary face contributions for Green-Gauss gradient computation.
 */
template <class Device>
struct green_gauss_boundary_face{
  typedef Device device_type;
  typedef typename ViewTypes<Device>::scalar_field_type scalar_field_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::face_cell_conn_type face_cell_conn_type;
  typedef typename ViewTypes<Device>::vector_field_type vector_field_type;
  typedef typename ViewTypes<Device>::gradient_storage_field_type gradient_storage_field_type;

  scalar_field_type cell_volumes_;
  face_cell_conn_type face_cell_conn_;
  face_cell_conn_type cell_flux_index_;
  solution_field_type cell_values_;
  gradient_storage_field_type cell_gradient_;
  vector_field_type face_normal_;

  green_gauss_boundary_face(Faces<Device> faces, solution_field_type cell_values, Cells<Device> cells):
    cell_volumes_(cells.volumes_),
    face_cell_conn_(faces.face_cell_conn_),
    cell_flux_index_(faces.cell_flux_index_),
    cell_values_(cell_values),
    cell_gradient_(cells.cell_gradient_),
    face_normal_(faces.face_normal_)
  {}

KOKKOS_INLINE_FUNCTION
void operator()( int i )const{
  int index = face_cell_conn_(i,0);

  double gradient[5][3];
  double conservatives[5];
  double primitives[5];

  for(int icomp = 0; icomp < 5; ++icomp)
    conservatives[icomp] = cell_values_(index,icomp);

  ComputePrimitives<device_type>(conservatives, primitives);

  for(int icomp = 0; icomp < 5; ++icomp)
  {
    for(int idir = 0; idir < 3; ++idir)
    {
    gradient[icomp][idir] = primitives[icomp]*face_normal_(i,idir);
    }
  }

#ifdef ATOMICS_FLUX
  for (int icomp = 0; icomp < 5; ++icomp)
  {
    for(int idir = 0; idir < 3; ++idir)
    {
      double * cell = &cell_gradient_(index,0,icomp,idir);
      Kokkos::atomic_fetch_add(cell, gradient[icomp][idir]/cell_volumes_(index));
    }
  }
#endif

#ifdef CELL_FLUX
for (int icomp = 0; icomp < 5; ++icomp)
{
  for(int idir = 0; idir < 3; ++idir)
  {
    cell_gradient_(index,cell_flux_index_(i,0),icomp, idir) = gradient[icomp][idir]/cell_volumes_(index);
  }
}
#endif
}

};

/* green_gauss_gradient_sum
 * functor to sum all of the contributions to the gradient
 * uses either gather-sum or atomics for thread safety.
 */
template <class Device>
struct green_gauss_gradient_sum{
  typedef Device device_type;
  typedef typename ViewTypes<Device>::gradient_storage_field_type gradient_storage_field_type;
  typedef typename ViewTypes<Device>::gradient_field_type gradient_field_type;
  gradient_storage_field_type face_gradient_;
  gradient_field_type gradient_;
  int number_faces_;

  green_gauss_gradient_sum(Cells<Device> cells, gradient_field_type gradient):
    face_gradient_(cells.cell_gradient_),
    gradient_(gradient),
    number_faces_(cells.nfaces_)
  {}

KOKKOS_INLINE_FUNCTION
void operator()( int i )const{

  for(int icomp = 0; icomp < 5; ++icomp)
  {
    
    for(int idir = 0; idir < 3; ++idir)
    {
      gradient_(i,icomp,idir) = 0;
    }
  }
#ifdef ATOMICS_FLUX
    for(int iface=0; iface<1; ++iface)
#else
    for(int iface=0; iface<number_faces_; ++iface)
#endif
  {

    for(int icomp = 0; icomp < 5; ++icomp)
    {
      for(int idir = 0; idir < 3; ++idir)
      {
        gradient_(i,icomp,idir) += face_gradient_(i,iface,icomp,idir);
      }
    }
  }
}
};

#endif


/*GreenGauss
 * contains all of the functions need to compute and communicate cell gradients.
 * aggregates the functors that are needed to compute and sum boundary and internal contributions
 * also communicates the cell gradient data on ghosted cells.
 */
template <class Device>
class GreenGauss {
  typedef typename ViewTypes<Device>::scalar_field_type scalar_field_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::gradient_field_type gradient_field_type;
  public:
    GreenGauss(Faces<Device> & internal_faces, std::vector<Faces<Device> *> & bc_faces, Cells<Device> & cells, struct MeshData<Device> & mesh_data, int total_send_count, int total_recv_count):
      internal_faces_(internal_faces),
      bc_faces_(bc_faces),
      cells_(cells),
      mesh_data_(mesh_data),
      ghosted_gradient_vars("ghosted_gradient_vars", total_recv_count*5*3),
      ghosted_gradient_vars_host(Kokkos::create_mirror(ghosted_gradient_vars)),
      shared_gradient_vars("shared_gradient_vars", total_send_count*5*3),
      shared_gradient_vars_host(Kokkos::create_mirror(shared_gradient_vars))
        {}

    //computes the gradient on locally owned cells.
    void compute_gradients(solution_field_type sol_np1_vec, gradient_field_type gradients){
      //Internal Faces
      const int ninternal_faces = internal_faces_.nfaces_;
      green_gauss_face<Device> face_gradient(internal_faces_, sol_np1_vec, cells_);
      parallel_for(ninternal_faces, face_gradient);
  
      //Boundary Faces
      typename std::vector<Faces<Device> *>::iterator bcf_iter, bcf_iter_end;
      bcf_iter = bc_faces_.begin();
      bcf_iter_end = bc_faces_.end();
      for(; bcf_iter != bcf_iter_end; ++bcf_iter){
        Faces<Device> * faces = *bcf_iter;
        const int nboundary_faces = faces->nfaces_;
        green_gauss_boundary_face<Device> bc_gradient(*faces, sol_np1_vec, cells_);
        parallel_for(nboundary_faces, bc_gradient);
      }
  
      //Sum of all contributions.
      green_gauss_gradient_sum<Device> gradient_sum(cells_, gradients);
      parallel_for(mesh_data_.num_owned_cells, gradient_sum);
      Device::fence();
    }

    //communicate the computed gradient for ghost cells.
    void communicate_gradients(gradient_field_type gradients){
      //copy values to be send from device to host
      extract_shared_tensor<Device, 5, 3> extract_shared_gradients(gradients, mesh_data_.send_local_ids, shared_gradient_vars);//sol_np1_vec, send_local_ids, shared_cells);
      parallel_for(mesh_data_.num_ghosts,extract_shared_gradients);
      Device::fence();
      Kokkos::deep_copy(shared_gradient_vars_host, shared_gradient_vars);
  
      communicate_ghosted_cell_data(mesh_data_.sendCount, mesh_data_.recvCount, shared_gradient_vars_host.ptr_on_device(),ghosted_gradient_vars_host.ptr_on_device(), 15);
  
      //copy values to be sent from host to device
      Kokkos::deep_copy(ghosted_gradient_vars, ghosted_gradient_vars_host);
      insert_ghost_tensor<Device, 5, 3> insert_ghost_gradients(gradients, mesh_data_.recv_local_ids, ghosted_gradient_vars);
      parallel_for(mesh_data_.num_ghosts, insert_ghost_gradients);
      Device::fence();
    }

  private:
    Faces<Device> & internal_faces_;
    std::vector<Faces<Device> *> & bc_faces_;
    Cells<Device> & cells_;
    struct MeshData<Device> & mesh_data_;
    scalar_field_type ghosted_gradient_vars;
    typename scalar_field_type::HostMirror ghosted_gradient_vars_host;
    scalar_field_type shared_gradient_vars;
    typename scalar_field_type::HostMirror shared_gradient_vars_host;
};
