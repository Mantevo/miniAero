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
#ifndef INCLUDE_FACES_H_
#define INCLUDE_FACES_H_

#include "Face.h"
#include <Kokkos_Core.hpp>
#include "ViewTypes.h"
#include <Kokkos_Sort.hpp>

/*Faces
 * struct containing the face data used for simulation.
 * Includes things such as coordinates, face_normal, and face to element
 * connectivity.
 */
template <class Device>
struct Faces{
   typedef Kokkos::View<int *, Device> id_type;
   typedef typename ViewTypes<Device>::vector_field_type vector_field_type;
   typedef typename ViewTypes<Device>::face_cell_conn_type face_cell_conn_type;

public:
   int nfaces_;
   int ncells_;
   vector_field_type coordinates_, face_normal_, face_tangent_, face_binormal_;
   face_cell_conn_type face_cell_conn_;
   face_cell_conn_type cell_flux_index_;
   id_type permute_vector_;

   Faces(){}

   Faces(int nfaces, int ncells) :
   nfaces_(nfaces),
   ncells_(ncells),
   coordinates_("coordinates", nfaces),
   face_normal_("face_normal",nfaces),
   face_tangent_("face_tangent", nfaces),
   face_binormal_("face_binormal", nfaces),
   face_cell_conn_("face_cell_conn", nfaces),
   cell_flux_index_("cell_flux_index", nfaces),
   permute_vector_("cell_permute_vector",nfaces)
   {
   }

};

template<class ViewType>
struct printfunctor {
  ViewType view;
  printfunctor(ViewType& v):view(v){}

  KOKKOS_INLINE_FUNCTION
  void operator() (int i) const {
    if(view(i)>=view.extent(0))
      printf("%i %i\n",i,view(i));
  }
};
/* copy_faces
 * functor to copy from host setup datastructure
 * to Kokkos datastructures.
 */
template <class Device>
void copy_faces(Faces<Device> device_faces, std::vector<Face> & mesh_faces){

  //Need host mirror
  typedef typename Kokkos::View<int *, Device>::HostMirror id_type;
  typedef typename ViewTypes<Device>::vector_field_type::HostMirror vector_field_type;
  typedef typename ViewTypes<Device>::face_cell_conn_type::HostMirror face_cell_conn_type;

  int nfaces = mesh_faces.size();
  vector_field_type coordinates = Kokkos::create_mirror(device_faces.coordinates_);
  vector_field_type face_normal = Kokkos::create_mirror(device_faces.face_normal_);
  vector_field_type face_tangent = Kokkos::create_mirror(device_faces.face_tangent_);
  vector_field_type face_binormal = Kokkos::create_mirror(device_faces.face_binormal_);

  face_cell_conn_type face_cell_conn = Kokkos::create_mirror(device_faces.face_cell_conn_);
  face_cell_conn_type cell_flux_index = Kokkos::create_mirror(device_faces.cell_flux_index_);

    double a_vec[3], t_vec[3], b_vec[3];
    for(int i = 0; i < mesh_faces.size(); ++i){
        face_cell_conn(i,0) = mesh_faces[i].GetElem1();
        face_cell_conn(i,1) = mesh_faces[i].GetElem2();
        cell_flux_index(i,0) = mesh_faces[i].GetElem1_FluxIndex();
        cell_flux_index(i,1) = mesh_faces[i].GetElem2_FluxIndex();

        const double * coords = mesh_faces[i].GetCoords();
        mesh_faces[i].GetAreaVecAndTangentVec(a_vec, t_vec, b_vec);
        for(int j = 0; j < 3; ++j){
            coordinates(i,j)   = coords[j];
            face_normal(i,j)   = a_vec[j];
            face_tangent(i,j)  = t_vec[j];
            face_binormal(i,j) = b_vec[j];
        }
    }
  Kokkos::deep_copy(device_faces.face_cell_conn_, face_cell_conn);
  Kokkos::deep_copy(device_faces.cell_flux_index_, cell_flux_index);
  Kokkos::deep_copy(device_faces.coordinates_, coordinates);
  Kokkos::deep_copy(device_faces.face_normal_, face_normal);
  Kokkos::deep_copy(device_faces.face_tangent_, face_tangent);
  Kokkos::deep_copy(device_faces.face_binormal_, face_binormal);

  if(device_faces.face_cell_conn_.extent(0) > 0) {
    typedef Kokkos::View<int *, Kokkos::LayoutStride, Device> view_type;
    typedef Kokkos::BinOp1D< view_type > CompType;
    view_type face_cell_left = Kokkos::subview(device_faces.face_cell_conn_,Kokkos::ALL(),0);

    typedef Kokkos::MinMax<int,Device> reducer_type;
    typedef typename reducer_type::value_type minmax_type;
    minmax_type minmax;
    Kokkos::parallel_reduce(face_cell_left.extent(0), KOKKOS_LAMBDA (const int& i, minmax_type& lminmax) {
      if(face_cell_left(i)<lminmax.min_val) lminmax.min_val = face_cell_left(i);
      if(face_cell_left(i)>lminmax.max_val) lminmax.max_val = face_cell_left(i);
    },reducer_type(minmax));

    Kokkos::BinSort<view_type, CompType, Device, int> bin_sort(face_cell_left,CompType(face_cell_left.extent(0)/2,minmax.min_val,minmax.max_val),true);
    bin_sort.create_permute_vector();
    Kokkos::deep_copy(device_faces.permute_vector_, bin_sort.sort_order);
  }
}

#endif
