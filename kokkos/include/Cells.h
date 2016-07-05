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
#ifndef INCLUDE_CELLS_H_
#define INCLUDE_CELLS_H_
#include "Cell.h"
#include <Kokkos_View.hpp>
#include "ViewTypes.h"


/*Cells
 * struct containing the cell data used for simulation.
 * Includes things such as coordinates, volume, the gradient, and
 * the flux contributions to the residual.
 */
template <class Device>
struct Cells{
  typedef typename ViewTypes<Device>::scalar_field_type scalar_field_type;
  typedef typename ViewTypes<Device>::vector_field_type vector_field_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::cell_face_conn_type cell_face_conn_type;
  typedef typename ViewTypes<Device>::cell_storage_field_type cell_storage_field_type;
  typedef typename ViewTypes<Device>::gradient_storage_field_type gradient_storage_field_type;

public:
   int ncells_;
   int nfaces_;
   vector_field_type coordinates_;
   scalar_field_type volumes_;
   cell_storage_field_type cell_flux_;
   gradient_storage_field_type cell_gradient_;

   Cells(){}

   Cells(int ncells, int faces_per_elem) :
   ncells_(ncells),
   nfaces_(faces_per_elem),
   coordinates_("cell_coordinates", ncells),
   volumes_("cell_volumes", ncells),
   cell_flux_("cell_flux", ncells, faces_per_elem), // Faces_per_elem needed for  gather-sum option.
   cell_gradient_("gradient", ncells, faces_per_elem)
   {
   }
};

/*zero_cell_flux
 * Functor to reset the flux contributions to the residual
 * to zero.
 */

template <class Device>
struct zero_cell_flux{

  typedef Device     device_type;
  typedef typename ViewTypes<Device>::cell_storage_field_type cell_storage_field_type;
  typedef typename ViewTypes<Device>::gradient_storage_field_type gradient_storage_field_type;

  const int ncells_;
  const int nfaces_;
  cell_storage_field_type cell_flux_;
  gradient_storage_field_type cell_gradient_;

  zero_cell_flux(Cells<Device> cells):
        ncells_(cells.ncells_),
        nfaces_(cells.nfaces_),
        cell_flux_(cells.cell_flux_),
        cell_gradient_(cells.cell_gradient_)
        {}

  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{
    for (int icomp = 0; icomp < 5; ++icomp) {
      for(int iface = 0; iface<nfaces_; ++iface) {
        cell_flux_(i,iface,icomp) = 0.0;
      }
      for(int iDir = 0; iDir < 3; ++iDir)
      {
        cell_gradient_(i,0,icomp,iDir) = 0.0;
      }
    }


  }
};

/*copy_cell_data
 * functor to copy the Cell information from the setup datastructure to
 * Kokkos datastructure.
 */
template <class Device>
void copy_cell_data(Cells<Device> device_cells, std::vector<Cell> & mesh_cells){
  
  typedef typename Kokkos::View<double *, Device>::HostMirror scalar_field_type;
  typedef typename Kokkos::View<double *[3], Device>::HostMirror vector_field_type;

  vector_field_type coordinates = Kokkos::create_mirror(device_cells.coordinates_);
  scalar_field_type volumes = Kokkos::create_mirror(device_cells.volumes_);

  int ncells = mesh_cells.size();

    for(int i = 0; i < mesh_cells.size(); ++i){
      volumes(i) = mesh_cells[i].GetVolume();
      for(int j=0; j<3; ++j){
        double * tmp_coord = mesh_cells[i].GetCoords();
        coordinates(i,j) = tmp_coord[j];
      }
    }
  Kokkos::deep_copy(device_cells.volumes_, volumes);
  Kokkos::deep_copy(device_cells.coordinates_, coordinates);
}

#endif
