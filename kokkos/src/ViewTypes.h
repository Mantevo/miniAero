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
#ifndef INCLUDE_VIEWTYPES_H_
#define INCLUDE_VIEWTYPES_H_

/*ViewTypes
 * struct that contains typedefs of commonly used Kokkos Views.
 */
template <class Device>
struct ViewTypes{
  typedef Kokkos::View<double *, Device> scalar_field_type;
  typedef Kokkos::View<double *[5], Device> solution_field_type;
  typedef Kokkos::View<int *[2], Device> face_cell_conn_type;
  typedef Kokkos::View<int **, Device> cell_face_conn_type;
  typedef Kokkos::View<double *[3], Kokkos::LayoutRight, Device> vector_field_type;
  typedef Kokkos::View<double **[5], Device> cell_storage_field_type;
  typedef Kokkos::View<double *[5][3], Device> gradient_field_type;
  typedef Kokkos::View<double **[5][3], Device> gradient_storage_field_type;

  typedef Kokkos::View<const double *, Device> c_scalar_field_type;
  typedef Kokkos::View<const double *[5], Device> c_solution_field_type;
  typedef Kokkos::View<const int *[2], Device> c_face_cell_conn_type;
  typedef Kokkos::View<const int **, Device> c_cell_face_conn_type;
  typedef Kokkos::View<const double *[3], Kokkos::LayoutRight, Device> c_vector_field_type;
  typedef Kokkos::View<const double **[5], Device> c_cell_storage_field_type;
  typedef Kokkos::View<const double *[5][3], Device> c_gradient_field_type;
  typedef Kokkos::View<const double **[5][3], Device> c_gradient_storage_field_type;

  typedef Kokkos::View<const double *, Device, Kokkos::MemoryTraits<Kokkos::RandomAccess> > c_rnd_scalar_field_type;
  typedef Kokkos::View<const double *[5], Device, Kokkos::MemoryTraits<Kokkos::RandomAccess> > c_rnd_solution_field_type;
  typedef Kokkos::View<const int *[2], Device, Kokkos::MemoryTraits<Kokkos::RandomAccess> > c_rnd_face_cell_conn_type;
  typedef Kokkos::View<const int **, Device, Kokkos::MemoryTraits<Kokkos::RandomAccess> > c_rnd_cell_face_conn_type;
  typedef Kokkos::View<const double *[3], Kokkos::LayoutRight, Device, Kokkos::MemoryTraits<Kokkos::RandomAccess> > c_rnd_vector_field_type;
  typedef Kokkos::View<const double **[5], Device, Kokkos::MemoryTraits<Kokkos::RandomAccess> > c_rnd_cell_storage_field_type;
  typedef Kokkos::View<const double *[5][3], Device, Kokkos::MemoryTraits<Kokkos::RandomAccess> > c_rnd_gradient_field_type;
  typedef Kokkos::View<const double **[5][3], Device, Kokkos::MemoryTraits<Kokkos::RandomAccess> > c_rnd_gradient_storage_field_type;


};

#endif
