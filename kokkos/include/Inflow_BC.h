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
#ifndef INCLUDE_INFLOW_BC_H_
#define INCLUDE_INFLOW_BC_H_

#include <cstdio>
#include <Kokkos_View.hpp>
#include "Faces.h"
#ifdef ATOMICS_FLUX
#include "Kokkos_Atomic.hpp"
#endif

/*compute_inflow_BC_flux
 * functor to compute contribution of inflow boundary condition
 * state is fixed and passed as flow_state
 * The state along with the internal cell state are used to compute
 * the external face flux contribution
 */
template<class Device, class FluxType>
struct compute_inflowBC_flux {
  typedef Device device_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::face_cell_conn_type face_cell_conn_type;
  typedef typename ViewTypes<Device>::cell_storage_field_type cell_storage_field_type;
  typedef typename ViewTypes<Device>::vector_field_type vector_field_type;

  face_cell_conn_type face_cell_conn_;
  face_cell_conn_type cell_flux_index_;
  solution_field_type cell_values_;
  cell_storage_field_type cell_flux_;
  vector_field_type face_normal_, face_tangent_, face_binormal_;
  FluxType flux_evaluator_;
  double flow_state_[5];

  compute_inflowBC_flux(Faces<Device> faces, solution_field_type cell_values,
      Cells<Device> cells, double * flow_state, FluxType flux) :
      face_cell_conn_(faces.face_cell_conn_), cell_flux_index_(
          faces.cell_flux_index_), cell_values_(cell_values), cell_flux_(
          cells.cell_flux_), face_normal_(faces.face_normal_), face_tangent_(
          faces.face_tangent_), face_binormal_(faces.face_binormal_), flux_evaluator_(
          flux) {
    for (int i = 0; i < 5; ++i)
      flow_state_[i] = flow_state[i];
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    int index = face_cell_conn_(i, 0);

    double flux[5];
    double conservatives_r[5];
    double conservatives_l[5];
    double primitives_r[5];
    double primitives_l[5];
    double normal[3];
    double tangent[3];
    double binormal[3];

    for (int icomp = 0; icomp < 5; ++icomp) {
      conservatives_l[icomp] = cell_values_(index, icomp);
      conservatives_r[icomp] = flow_state_[icomp];
    }

    ComputePrimitives<device_type>(conservatives_l, primitives_l);
    ComputePrimitives<device_type>(conservatives_r, primitives_r);
    for (int icomp = 0; icomp < 3; ++icomp) {
      normal[icomp] = face_normal_(i, icomp);
      tangent[icomp] = face_tangent_(i, icomp);
      binormal[icomp] = face_binormal_(i, icomp);
    }

    flux_evaluator_.compute_flux(primitives_l, primitives_r, flux, normal,
        tangent, binormal);

#ifdef ATOMICS_FLUX
    for (int icomp = 0; icomp < 5; ++icomp)
    {

      double * cell = &cell_flux_(index,0,icomp);
      Kokkos::atomic_add(cell, -flux[icomp]);
    }
#endif

#ifdef CELL_FLUX
    for (int icomp = 0; icomp < 5; ++icomp)
    {
      cell_flux_(index,cell_flux_index_(i,0),icomp) = -flux[icomp];
    }
#endif

  }
};

#endif
