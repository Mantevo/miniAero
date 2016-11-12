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
 * CopyGhost.h
 *
 *  Created on: Jan 9, 2014
 *      Author: kjfrank
 */

#ifndef COPYGHOST_H_
#define COPYGHOST_H_

#include <Kokkos_Core.hpp>
#include <vector>
/* extract_shared_vector
 * functor to copy ghosted vector data from full array to buffer to send to other processors.
 */
template <class Device>
struct extract_shared_vector_1d{
  typedef Device     device_type;
  typedef Kokkos::View<double *, device_type> value_type;
  typedef Kokkos::View<double *, device_type> flat_type;
  typedef Kokkos::View<int *, device_type> id_map_type;

  value_type cell_values_;
  id_map_type local_id_map_;
  flat_type  shared_cells_;
  extract_shared_vector_1d(value_type cell_value,
      id_map_type local_id_map,
      flat_type shared_cells):
        cell_values_(cell_value),
        local_id_map_(local_id_map),
        shared_cells_(shared_cells){}

  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{
    shared_cells_(i) = cell_values_(local_id_map_(i));
  }
};

template <class Device>
struct insert_ghost_vector_1d{
  typedef Device     device_type;
  typedef Kokkos::View<double *, device_type> value_type;
  typedef Kokkos::View<double *, device_type> flat_type;
  typedef Kokkos::View<int *, device_type> id_map_type;

  value_type cell_values_;
  id_map_type local_id_map_;
  flat_type  shared_cells_;
  insert_ghost_vector_1d(value_type cell_value,
      id_map_type local_id_map,
      flat_type shared_cells):
        cell_values_(cell_value),
        local_id_map_(local_id_map),
        shared_cells_(shared_cells){}

  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{
    cell_values_(local_id_map_(i)) = shared_cells_(i);
  }
};

/* extract_shared_vector
 * functor to copy ghosted vector data from full array to buffer to send to other processors.
 */
template <class Device, int dim>
struct extract_shared_vector{
  typedef Device     device_type;
  typedef Kokkos::View<double *[dim], device_type> conservative_type;
  typedef Kokkos::View<double *, device_type> flat_type;
  typedef Kokkos::View<int *, device_type> id_map_type;

  conservative_type cell_values_;
  id_map_type local_id_map_;
  flat_type  shared_cells_;
  extract_shared_vector(conservative_type cell_value,
      id_map_type local_id_map,
      flat_type shared_cells):
        cell_values_(cell_value),
        local_id_map_(local_id_map),
        shared_cells_(shared_cells){}

  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{
    int index = local_id_map_(i);
    for(int j=0; j<dim; ++j){
      shared_cells_(i*dim+j) = cell_values_(index,j);
    }
  }
};

/*insert_ghost_vector
 * functor to copy vector ghosted data from buffer to full array from other processors.
 */

template <class Device, int dim>
struct insert_ghost_vector{
  typedef Device     device_type;
  typedef Kokkos::View<double *[dim], Device> conservative_type;
  typedef Kokkos::View<double *, Device> flat_type;
  typedef Kokkos::View<int *, Device> id_map_type;

  conservative_type cell_values_;
  id_map_type local_id_map_;
  flat_type ghosted_cells_;
  insert_ghost_vector(conservative_type cell_value,
      id_map_type local_id_map,
      flat_type ghosted_cells):
        cell_values_(cell_value),
        local_id_map_(local_id_map),
        ghosted_cells_(ghosted_cells){}
  insert_ghost_vector()
{}
  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{
    int index = local_id_map_(i);
    for(int j=0; j<dim; ++j){
      cell_values_(index,j) = ghosted_cells_(i*dim+j);
    }
  }
};

/* extract_shared_tesnor
 * functor to copy ghosted tensor data from full array to buffer to send to other processors.
 */
template <class Device, int dim1, int dim2>
struct extract_shared_tensor{
  typedef Device     device_type;
  typedef Kokkos::View<double *[dim1][dim2], device_type> gradient_type;
  typedef Kokkos::View<double *, device_type> flat_type;
  typedef Kokkos::View<int *, device_type> id_map_type;

  gradient_type cell_values_;
  id_map_type local_id_map_;
  flat_type  shared_cells_;
  extract_shared_tensor(gradient_type cell_value,
      id_map_type local_id_map,
      flat_type shared_cells):
        cell_values_(cell_value),
        local_id_map_(local_id_map),
        shared_cells_(shared_cells){}

  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{
    int index = local_id_map_(i);
    for(int j=0; j<dim1; ++j){
      for(int iDir=0; iDir<dim2; ++iDir)
      {
        shared_cells_(i*dim1*dim2+j*dim2+iDir) = cell_values_(index,j,iDir);
      }
    }
  }
};

/*insert_ghost_tensor
 * functor to copy tensor ghosted data from buffer to full array from other processors.
 */
template <class Device, int dim1, int dim2>
struct insert_ghost_tensor{
  typedef Device     device_type;
  typedef Kokkos::View<double *[dim1][dim2], Device> gradient_type;
  typedef Kokkos::View<double *, Device> flat_type;
  typedef Kokkos::View<int *, Device> id_map_type;

  gradient_type cell_values_;
  id_map_type local_id_map_;
  flat_type ghosted_cells_;
  insert_ghost_tensor(gradient_type cell_value,
      id_map_type local_id_map,
      flat_type ghosted_cells):
        cell_values_(cell_value),
        local_id_map_(local_id_map),
        ghosted_cells_(ghosted_cells){}
  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{
    int index = local_id_map_(i);
    for(int j=0; j<dim1; ++j){
      for(int iDir=0; iDir<dim2; ++iDir)
      {
      cell_values_(index,j,iDir) = ghosted_cells_(i*dim1*dim2+j*dim2+iDir);
      }
    }
  }
};

/*communicate_ghosted_cell_data
 * function to communicate buffered data between processors that share ghosted data.
 */
void communicate_ghosted_cell_data(std::vector<int> & sendCount, std::vector<int> & recvCount,
    double *send_data, double *recv_data, int data_per_cell);


#endif /* COPYGHOST_H_ */
