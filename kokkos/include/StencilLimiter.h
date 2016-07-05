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
 * StencilLimiter.h
 *
 *  Created on: May 27, 2014
 *      Author: kjfrank
 */
#include <cstdio>
#include <Kokkos_View.hpp>
#include "Faces.h"
#include "MathToolsDevice.h"
#include "VenkatLimiter.h"
#include "ViewTypes.h"

#ifdef ATOMICS_FLUX
#include "Kokkos_Atomic.hpp"
#endif

#ifndef STENCILLIMITER_H_
#define STENCILLIMITER_H_

/* min_max_face
 * functor to compute the minimum and maximum value at each face
 * and scatters to the 2 connected elements.
 */
template <class Device, bool interior>
struct min_max_face{
  typedef Device device_type;
  typedef typename ViewTypes<Device>::scalar_field_type scalar_field_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::face_cell_conn_type face_cell_conn_type;
  typedef typename ViewTypes<Device>::vector_field_type vector_field_type;
  typedef typename ViewTypes<Device>::cell_storage_field_type cell_storage_field_type;

  scalar_field_type cell_volumes_;
  face_cell_conn_type face_cell_conn_;
  face_cell_conn_type cell_flux_index_;
  solution_field_type cell_values_;
  vector_field_type face_normal_;
  cell_storage_field_type stencil_min_, stencil_max_;

  min_max_face(Faces<Device> faces, solution_field_type cell_values, Cells<Device> cells,
     cell_storage_field_type stencil_min, cell_storage_field_type stencil_max):
    face_cell_conn_(faces.face_cell_conn_),
    cell_flux_index_(faces.cell_flux_index_),
    cell_values_(cell_values),
    stencil_min_(stencil_min),
    stencil_max_(stencil_max)
  {}

KOKKOS_INLINE_FUNCTION
void operator()( int i )const{
  int left_index = face_cell_conn_(i,0);
  int right_index = face_cell_conn_(i,1);

	double conservatives_l[5];
	double conservatives_r[5];
  double primitives_l[5];
  double primitives_r[5];

  for (int icomp = 0; icomp < 5; ++icomp)
  {
    if(interior){ 
      conservatives_l[icomp] = cell_values_(left_index,icomp);
      conservatives_r[icomp] = cell_values_(right_index,icomp);
    }
    else{
      conservatives_l[icomp] = cell_values_(left_index,icomp);
    }
  }

  if(interior){
    ComputePrimitives<device_type>(conservatives_l, primitives_l); 
    ComputePrimitives<device_type>(conservatives_r, primitives_r); 
  }
  else{
    ComputePrimitives<device_type>(conservatives_l, primitives_l); 
  }

  for (int icomp = 0; icomp < 5; ++icomp)
  {
    double face_min, face_max;
    if(interior){ 
      face_min = MathTools<device_type>::min(primitives_r[icomp],primitives_l[icomp]);
      face_max = MathTools<device_type>::max(primitives_r[icomp],primitives_l[icomp]);
    }
    else{
      face_min = primitives_l[icomp];
      face_max = primitives_l[icomp];
    }

#ifdef ATOMICS_FLUX
    //Need compare and exhange here instead of atomic add

    double * left_cell_min = &stencil_min_(left_index,0,icomp);
    bool success=false;
    do{
      double old_left_min =  *left_cell_min;
      double new_left_min = MathTools<device_type>::min(*left_cell_min, face_min);
      double new_value = Kokkos::atomic_compare_exchange(left_cell_min, old_left_min, new_left_min);
      success = new_value == new_left_min;
    } while(!success);
    double * left_cell_max = &stencil_max_(left_index,0,icomp);
    success=false;
    do{
      double old_left_max =  *left_cell_max;
      double new_left_max = MathTools<device_type>::max(*left_cell_max, face_max);
      double new_value = Kokkos::atomic_compare_exchange(left_cell_max, old_left_max, new_left_max);
      success = new_value == new_left_max;
    } while(!success);

    if(interior){
      double * right_cell_min = &stencil_min_(right_index,0,icomp);
      success=false;
      do{
        double old_right_min =  *right_cell_min;
        double new_right_min = MathTools<device_type>::min(*right_cell_min, face_min);
        double new_value = Kokkos::atomic_compare_exchange(right_cell_min, old_right_min, new_right_min);
        success = new_value == new_right_min;
      } while(!success);
      double * right_cell_max = &stencil_max_(right_index,0,icomp);
      success=false;
      do{
        double old_right_max =  *right_cell_max;
        double new_right_max = MathTools<device_type>::max(*right_cell_max, face_max);
        double new_value = Kokkos::atomic_compare_exchange(right_cell_max, old_right_max, new_right_max);
        success = new_value == new_right_max;
      } while(!success);
    }
#endif

#ifdef CELL_FLUX
    stencil_min_(left_index, cell_flux_index_(i,0), icomp) = face_min;
    stencil_max_(left_index, cell_flux_index_(i,0), icomp) = face_max;

    if(interior){
      stencil_min_(right_index, cell_flux_index_(i,1), icomp) = face_min;
      stencil_max_(right_index, cell_flux_index_(i,1), icomp) = face_max;
    }
#endif
  }
}
};

/* initialize_min_max
 * functor that initializes the minimum and maximum values using very large or very small numbers.
 */
template <class Device>
struct initialize_min_max{

  typedef Device     device_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::cell_storage_field_type cell_storage_field_type;

  const int nfaces_;
  solution_field_type stencil_min_, stencil_max_;
  cell_storage_field_type stored_min_, stored_max_;

  initialize_min_max(int nfaces, solution_field_type stencil_min, solution_field_type stencil_max, cell_storage_field_type stored_min, cell_storage_field_type stored_max):
        nfaces_(nfaces),
        stencil_min_(stencil_min),
        stencil_max_(stencil_max),
        stored_min_(stored_min),
        stored_max_(stored_max)
        {}

  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{
    for (int icomp = 0; icomp < 5; ++icomp) {
      stencil_min_(i,icomp) = 1.0e300;
      stencil_max_(i,icomp) = -1.0e300;
      for(int iface = 0; iface<nfaces_; ++iface) {
        stored_min_(i,iface,icomp) = 1.0e300;
        stored_max_(i,iface,icomp) = -1.0e300;
      }
    }
  }
};

/* gather_min_max
 * functor that computes the minimum or maximum value of each variable over the stencil
 * of each cell.  The stencil consists of all cells which share a face with
 * this cell.
 */

template <class Device>
struct gather_min_max{

  typedef Device     device_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::cell_storage_field_type cell_storage_field_type;

  const int ncells_;
  const int nfaces_;
  cell_storage_field_type stored_min_, stored_max_;
  solution_field_type stencil_min_, stencil_max_;

  gather_min_max(Cells<Device> cells, cell_storage_field_type stored_min, cell_storage_field_type stored_max, solution_field_type stencil_min, solution_field_type stencil_max):
        ncells_(cells.ncells_),
        nfaces_(cells.nfaces_),
        stored_min_(stored_min),
        stored_max_(stored_max),
        stencil_min_(stencil_min),
        stencil_max_(stencil_max)
        {}

  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{
    for (int icomp = 0; icomp < 5; ++icomp) {
#ifdef ATOMICS_FLUX
      stencil_min_(i,icomp) = stored_min_(i,0,icomp);
      stencil_max_(i,icomp) = stored_max_(i,0,icomp);
#endif

#ifdef CELL_FLUX
      for(int iface = 0; iface<nfaces_; ++iface) {
        stencil_min_(i,icomp) = MathTools<device_type>::min(stencil_min_(i,icomp),stored_min_(i,iface,icomp));
        stencil_max_(i,icomp) = MathTools<device_type>::max(stencil_max_(i,icomp),stored_max_(i,iface,icomp));
      }
#endif
    }
  }
};

/* initialize_limiter
 * functor that initializes the cell limiter value to 1.0.
 */

template <class Device>
struct initialize_limiter{

  typedef Device     device_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::cell_storage_field_type cell_storage_field_type;

  const int nfaces_;
  solution_field_type limiter_;
  cell_storage_field_type stored_limiter_;

  initialize_limiter(int nfaces, cell_storage_field_type stored_limiter, solution_field_type limiter):
        nfaces_(nfaces),
        stored_limiter_(stored_limiter),
        limiter_(limiter)
        {}

  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{
    for (int icomp = 0; icomp < 5; ++icomp) {
      for(int iface = 0; iface<nfaces_; ++iface) {
        stored_limiter_(i,iface,icomp) = 1.0;
      }
      limiter_(i,icomp) = 1.0;
    }
  }
};

/* gather_limiter
 * functor that gathers and takes the minimum limiter value of the connected faces.
 */
template <class Device>
struct gather_limiter{

  typedef Device     device_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::cell_storage_field_type cell_storage_field_type;

  const int nfaces_;
  cell_storage_field_type stored_limiter_; 
  solution_field_type limiter_;

  gather_limiter(int nfaces, cell_storage_field_type stored_limiter, solution_field_type limiter):
        nfaces_(nfaces),
        stored_limiter_(stored_limiter),
        limiter_(limiter)
        {}

  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{
    for (int icomp = 0; icomp < 5; ++icomp) {
#ifdef ATOMICS_FLUX
      limiter_(i,icomp) = stored_limiter_(i,0,icomp);
#endif

#ifdef CELL_FLUX
      for(int iface = 0; iface<nfaces_; ++iface) {
        limiter_(i,icomp) = MathTools<device_type>::min(limiter_(i,icomp),stored_limiter_(i,iface,icomp));
      }
#endif
    }
  }
};

/* limiter_face
 * functor that computes the limiter value for each face and scatter contribution
 * to the connected elements.  Uses gather-sum or atomics for thread safety.
 */
template <class Device, bool interior>
struct limiter_face{
  typedef Device device_type;
  typedef typename ViewTypes<Device>::scalar_field_type scalar_field_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::face_cell_conn_type face_cell_conn_type;
  typedef typename ViewTypes<Device>::vector_field_type vector_field_type;
  typedef typename ViewTypes<Device>::cell_storage_field_type cell_storage_field_type;
  typedef typename ViewTypes<Device>::gradient_field_type gradient_field_type;

  scalar_field_type cell_volumes_;
  face_cell_conn_type face_cell_conn_;
  face_cell_conn_type cell_flux_index_;
  solution_field_type cell_min_, cell_max_, cell_values_;
  vector_field_type face_coordinates_, cell_coordinates_;
  gradient_field_type cell_gradients_;
  cell_storage_field_type limiter_; 

  limiter_face(Faces<Device> faces, solution_field_type cell_values, Cells<Device> cells,
    gradient_field_type gradients,
    solution_field_type cell_min, solution_field_type cell_max, cell_storage_field_type limiter):
    face_cell_conn_(faces.face_cell_conn_),
    cell_flux_index_(faces.cell_flux_index_),
    cell_min_(cell_min),
    cell_max_(cell_max),
    cell_values_(cell_values),
    face_coordinates_(faces.coordinates_),
    cell_coordinates_(cells.coordinates_),
    cell_gradients_(gradients),
    limiter_(limiter)
  {}

KOKKOS_INLINE_FUNCTION
void operator()( int i )const{
  int left_index = face_cell_conn_(i,0);
  int right_index = face_cell_conn_(i,1);
	
  double conservatives_l[5];
	double conservatives_r[5];
  double primitives_l[5];
  double primitives_r[5];

  for (int icomp = 0; icomp < 5; ++icomp)
  {
    if(interior){ 
      conservatives_l[icomp] = cell_values_(left_index,icomp);
      conservatives_r[icomp] = cell_values_(right_index,icomp);
    }
    else{
      conservatives_l[icomp] = cell_values_(left_index,icomp);
    }
  }

  if(interior){
    ComputePrimitives<device_type>(conservatives_l, primitives_l); 
    ComputePrimitives<device_type>(conservatives_r, primitives_r); 
  }
  else{
    ComputePrimitives<device_type>(conservatives_l, primitives_l); 
  }

//Compute left limiter value and compute right limiter value

  double limiter_left[5], limiter_right[5];
  //compute displacement and distance from cell center to face center.
  double displacement_l[3];
  double displacement_r[3];
  double distance_l = 0;
  double distance_r = 0;
  for(int idir = 0; idir < 3; ++idir){
    displacement_l[idir] = face_coordinates_(i, idir)-cell_coordinates_(left_index, idir);
    distance_l += displacement_l[idir]*displacement_l[idir];
    if(interior){
      displacement_r[idir] = face_coordinates_(i, idir)-cell_coordinates_(right_index, idir);
      distance_r += displacement_r[idir]*displacement_r[idir];
      }
  }

  double dU_l[5];
  double dU_r[5];
  //Extrapolation
  for(int icomp = 0; icomp < 5; ++icomp){
    dU_l[icomp] = 0;
    dU_r[icomp] = 0;
    for(int idir = 0; idir < 3; ++idir){
      dU_l[icomp] += displacement_l[idir]*cell_gradients_(left_index, icomp, idir);
      if(interior)
        dU_r[icomp] += displacement_r[idir]*cell_gradients_(right_index, icomp, idir);
    }
  }


  for(int icomp = 0; icomp < 5; ++icomp){
    double dumax_l = cell_max_(left_index, icomp) - primitives_l[icomp];
    double dumin_l = cell_min_(left_index, icomp) - primitives_l[icomp];

    limiter_left[icomp] = VenkatLimiter<device_type>::limit(dumax_l, dumin_l, dU_l[icomp], distance_l);
    if(interior){
      double dumax_r = cell_max_(right_index, icomp) - primitives_r[icomp];
      double dumin_r = cell_min_(right_index, icomp) - primitives_r[icomp];
      limiter_right[icomp] = VenkatLimiter<device_type>::limit(dumax_r, dumin_r, dU_r[icomp], distance_r);
    }
  }

//Then write to memory
#ifdef ATOMICS_FLUX
  for (int icomp = 0; icomp < 5; ++icomp)
  {
    double * left_cell_limiter = &limiter_(left_index,0,icomp);
    bool success=false;
    do{
      double old_left_limiter =  *left_cell_limiter;
      double new_left_limiter = MathTools<device_type>::min(*left_cell_limiter, limiter_left[icomp]);
      double new_value = Kokkos::atomic_compare_exchange(left_cell_limiter, old_left_limiter, new_left_limiter);
      success = new_value == new_left_limiter;
    } while(!success);

    if(interior){
      double * right_cell_limiter = &limiter_(right_index,0,icomp);
      success=false;
      do{
        double old_right_limiter =  *right_cell_limiter;
        double new_right_limiter = MathTools<device_type>::min(*right_cell_limiter, limiter_right[icomp]);
        double new_value = Kokkos::atomic_compare_exchange(right_cell_limiter, old_right_limiter, new_right_limiter);
        success = new_value == new_right_limiter;
      } while(!success);
    }
  }
#endif

#ifdef CELL_FLUX
for (int icomp = 0; icomp < 5; ++icomp)
{
  limiter_(left_index, cell_flux_index_(i,0), icomp) = limiter_left[icomp]; 

  if(interior){
    limiter_(right_index, cell_flux_index_(i,1), icomp) = limiter_right[icomp]; 
  }
}
#endif
}
};

/*StencilLimiter
 * Class to compute cell value of the stencil limiter.
 * Has methods to compute the min/max value of stencil
 * and compute the limiter value for each cell.
 */
template <class Device>
class StencilLimiter{
  typedef typename ViewTypes<Device>::scalar_field_type scalar_field_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::cell_storage_field_type cell_storage_field_type;
  typedef typename ViewTypes<Device>::gradient_field_type gradient_field_type;

  public:
    StencilLimiter(Faces<Device> & internal_faces, std::vector<Faces<Device> *> & bc_faces, Cells<Device> & cells, struct MeshData<Device> & mesh_data, int total_send_count, int total_recv_count):
      internal_faces_(internal_faces),
      bc_faces_(bc_faces),
      cells_(cells),
      mesh_data_(mesh_data),
      ghosted_vars("ghosted_vars", total_recv_count*5),
      ghosted_vars_host(Kokkos::create_mirror(ghosted_vars)),
      shared_vars("shared_vars", total_send_count*5),
      shared_vars_host(Kokkos::create_mirror(shared_vars)),
      stored_min_("stored_min", cells.ncells_*5, cells.nfaces_),
      stored_max_("stored_max", cells.ncells_*5, cells.nfaces_),
      stored_limiter_("stored_limiter", cells.ncells_*5, cells.nfaces_),
      stencil_min_("stencil_min", cells.ncells_*5),
      stencil_max_("stencil_max", cells.ncells_*5)
        {}

  void compute_min_max(solution_field_type sol_np1_vec) {

    initialize_min_max<Device> init_min_max(cells_.nfaces_, stencil_min_, stencil_max_, stored_min_, stored_max_);
    parallel_for(mesh_data_.num_owned_cells, init_min_max);

    //Internal Faces
    const int ninternal_faces = internal_faces_.nfaces_;
    min_max_face<Device, true> min_max_internal(internal_faces_, sol_np1_vec, cells_, stored_min_, stored_max_);
    parallel_for(ninternal_faces, min_max_internal);
  
    //Boundary Faces
    typename std::vector<Faces<Device> *>::iterator bcf_iter, bcf_iter_end;
    bcf_iter = bc_faces_.begin();
    bcf_iter_end = bc_faces_.end();
    for(; bcf_iter != bcf_iter_end; ++bcf_iter){
      Faces<Device> * faces = *bcf_iter;
      const int nboundary_faces = faces->nfaces_;
      min_max_face<Device, false> bc_min_max(*faces, sol_np1_vec, cells_, stored_min_, stored_max_);
      parallel_for(nboundary_faces, bc_min_max);
    }
    Device::fence();
  
    gather_min_max<Device> gather(cells_, stored_min_, stored_max_, stencil_min_, stencil_max_);
    parallel_for(mesh_data_.num_owned_cells, gather);
    Device::fence();
  }

  void communicate_min_max(){

  // For min
      extract_shared_vector<Device, 5> extract_shared_min(stencil_min_, mesh_data_.send_local_ids, shared_vars);
      parallel_for(mesh_data_.num_ghosts, extract_shared_min);
      Device::fence();
      Kokkos::deep_copy(shared_vars_host, shared_vars);
  
      communicate_ghosted_cell_data(mesh_data_.sendCount, mesh_data_.recvCount, shared_vars_host.ptr_on_device(),ghosted_vars_host.ptr_on_device(), 5);
  
      Kokkos::deep_copy(ghosted_vars, ghosted_vars_host);
      insert_ghost_vector<Device, 5> insert_ghost_min(stencil_min_, mesh_data_.recv_local_ids, ghosted_vars);
      parallel_for(mesh_data_.num_ghosts, insert_ghost_min);
      Device::fence();

  // For max
      extract_shared_vector<Device, 5> extract_shared_max(stencil_max_, mesh_data_.send_local_ids, shared_vars);
      parallel_for(mesh_data_.num_ghosts, extract_shared_max);
      Device::fence();
      Kokkos::deep_copy(shared_vars_host, shared_vars);
  
      communicate_ghosted_cell_data(mesh_data_.sendCount, mesh_data_.recvCount, shared_vars_host.ptr_on_device(),ghosted_vars_host.ptr_on_device(), 5);
  
      Kokkos::deep_copy(ghosted_vars, ghosted_vars_host);
      insert_ghost_vector<Device, 5> insert_ghost_max(stencil_max_, mesh_data_.recv_local_ids, ghosted_vars);
      parallel_for(mesh_data_.num_ghosts, insert_ghost_max);
      Device::fence();
  // TODO: Maybe combined or overlapped in future.
  }

  void compute_limiter(solution_field_type sol_np1_vec, solution_field_type limiter, gradient_field_type gradients) {
    initialize_limiter<Device> init_limiter(cells_.nfaces_, stored_limiter_, limiter);
    parallel_for(mesh_data_.num_owned_cells, init_limiter);

    //Internal Faces
    const int ninternal_faces = internal_faces_.nfaces_;
    limiter_face<Device, true> limiter_internal(internal_faces_, sol_np1_vec, cells_, gradients,
      stencil_min_, stencil_max_, stored_limiter_);
    parallel_for(ninternal_faces, limiter_internal);
  
    //Boundary Faces
    typename std::vector<Faces<Device> *>::iterator bcf_iter, bcf_iter_end;
    bcf_iter = bc_faces_.begin();
    bcf_iter_end = bc_faces_.end();
    for(; bcf_iter != bcf_iter_end; ++bcf_iter){
      Faces<Device> * faces = *bcf_iter;
      const int nboundary_faces = faces->nfaces_;
      limiter_face<Device, false> limiter_bc(*faces, sol_np1_vec, cells_, gradients, stencil_min_, stencil_max_, stored_limiter_);
      parallel_for(nboundary_faces, limiter_bc);
    }
    Device::fence();
  
    gather_limiter<Device> gather(cells_.nfaces_, stored_limiter_, limiter);
    parallel_for(mesh_data_.num_owned_cells, gather);
    Device::fence();
  }

  void communicate_limiter(solution_field_type limiter) {

      extract_shared_vector<Device, 5> extract_shared_limiter(limiter, mesh_data_.send_local_ids, shared_vars);
      parallel_for(mesh_data_.num_ghosts, extract_shared_limiter);
      Device::fence();
      Kokkos::deep_copy(shared_vars_host, shared_vars);
  
      communicate_ghosted_cell_data(mesh_data_.sendCount, mesh_data_.recvCount, shared_vars_host.ptr_on_device(), ghosted_vars_host.ptr_on_device(), 5);
  
      Kokkos::deep_copy(ghosted_vars, ghosted_vars_host);
      insert_ghost_vector<Device, 5> insert_ghost_limiter(limiter, mesh_data_.recv_local_ids, ghosted_vars);
      parallel_for(mesh_data_.num_ghosts, insert_ghost_limiter);
      Device::fence();
  }

  private:
    Faces<Device> & internal_faces_;
    std::vector<Faces<Device> *> & bc_faces_;
    Cells<Device> & cells_;
    struct MeshData<Device> & mesh_data_;
    scalar_field_type ghosted_vars;
    typename scalar_field_type::HostMirror ghosted_vars_host;
    scalar_field_type shared_vars;
    typename scalar_field_type::HostMirror shared_vars_host;
    cell_storage_field_type stored_min_;
    cell_storage_field_type stored_max_;
    cell_storage_field_type stored_limiter_;
    solution_field_type stencil_min_;
    solution_field_type stencil_max_;
};

#endif
