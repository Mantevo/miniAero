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
#ifndef INCLUDE_NOSLIP_BC_H_
#define INCLUDE_NOSLIP_BC_H_

#include <cstdio>
#include <cmath>
#include <Kokkos_View.hpp>
#include "Faces.h"
#include "GasModel.h"

#ifdef ATOMICS_FLUX
#include "Kokkos_Atomic.hpp"
#endif

/*compuate_NoSlipBC_flux
 * functor to compute the contribution of a noSlip wall boundary condition
 * Sets the state for the flux evaluation such that the velocity at the
 * wall is zero.
 */
template<class Device, class InviscidFluxType, class ViscousFluxType>
struct compute_NoSlipBC_flux {
  typedef Device device_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::face_cell_conn_type face_cell_conn_type;
  typedef typename ViewTypes<Device>::vector_field_type vector_field_type;
  typedef typename ViewTypes<Device>::cell_storage_field_type cell_storage_field_type;

  face_cell_conn_type face_cell_conn_;
  face_cell_conn_type cell_flux_index_;
  solution_field_type cell_values_;
  cell_storage_field_type cell_flux_;
  vector_field_type cell_coordinates_;
  vector_field_type face_coordinates_, face_normal_, face_tangent_,
      face_binormal_;
  InviscidFluxType inviscid_flux_evaluator_;
  ViscousFluxType viscous_flux_evaluator_;

  compute_NoSlipBC_flux(Faces<Device> faces, solution_field_type cell_values,
      Cells<Device> cells, InviscidFluxType inviscid_flux,
      ViscousFluxType viscous_flux) :
      face_cell_conn_(faces.face_cell_conn_), cell_flux_index_(
          faces.cell_flux_index_), cell_values_(cell_values), cell_flux_(
          cells.cell_flux_), cell_coordinates_(cells.coordinates_), face_coordinates_(
          faces.coordinates_), face_normal_(faces.face_normal_), face_tangent_(
          faces.face_tangent_), face_binormal_(faces.face_binormal_), inviscid_flux_evaluator_(
          inviscid_flux), viscous_flux_evaluator_(viscous_flux) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    int index = face_cell_conn_(i, 0);

    double iflux[5];
    double vflux[5];
    double conservatives[5];
    double primitives_r[5];
    double primitives_l[5];
    double normal[3];
    double tangent[3];
    double binormal[3];

    for (int icomp = 0; icomp < 5; ++icomp) {
      conservatives[icomp] = cell_values_(index, icomp);
      vflux[icomp] = 0.0;
    }

    ComputePrimitives<device_type>(conservatives, primitives_l);

    for (int icomp = 0; icomp < 3; ++icomp) {
      normal[icomp] = face_normal_(i, icomp);
      tangent[icomp] = face_tangent_(i, icomp);
      binormal[icomp] = face_binormal_(i, icomp);
    }

    //scale normal since it includes area.
    double area_norm = 0;
    for (int icomp = 0; icomp < 3; ++icomp) {
      area_norm += normal[icomp] * normal[icomp];
    }
    area_norm = std::sqrt(area_norm);

    double uboundary = 0.0;
    uboundary += primitives_l[1] * normal[0] / area_norm;
    uboundary += primitives_l[2] * normal[1] / area_norm;
    uboundary += primitives_l[3] * normal[2] / area_norm;

    primitives_r[0] = primitives_l[0];
    primitives_r[1] = primitives_l[1] - 2 * uboundary * normal[0] / area_norm;
    primitives_r[2] = primitives_l[2] - 2 * uboundary * normal[1] / area_norm;
    primitives_r[3] = primitives_l[3] - 2 * uboundary * normal[2] / area_norm;
    primitives_r[4] = primitives_l[4];

    inviscid_flux_evaluator_.compute_flux(primitives_l, primitives_r, iflux,
        normal, tangent, binormal);

    if (ViscousFluxType::isViscous) {
      double primitives_face[5];
      double gradients_face[5][3];
      double distance_to_wall = 0;
      double unit_normal[3];
      primitives_face[0] = primitives_l[0];
      primitives_face[1] = 0.0;
      primitives_face[2] = 0.0;
      primitives_face[3] = 0.0;
      primitives_face[4] = primitives_l[4];
      for (int idir = 0; idir < 3; ++idir) {
        distance_to_wall += std::pow(
            face_coordinates_(i, idir) - cell_coordinates_(index, idir), 2);
        unit_normal[idir] = normal[idir] / area_norm;
      }
      double inv_distance_to_wall = 1.0 / std::sqrt(distance_to_wall);

      for (int icomp = 0; icomp < 5; ++icomp) {
        for (int idir = 0; idir < 3; ++idir) {
          gradients_face[icomp][idir] = (primitives_face[icomp]
              - primitives_l[icomp]) * unit_normal[idir] * inv_distance_to_wall;
        }
      }
      viscous_flux_evaluator_.compute_flux(gradients_face, primitives_face,
          normal, vflux);
    }

#ifdef ATOMICS_FLUX
    for (int icomp = 0; icomp < 5; ++icomp)
    {
      double * cell = &cell_flux_(index,0,icomp);
      Kokkos::atomic_add(cell, -iflux[icomp]+vflux[icomp]);
    }
#endif

#ifdef CELL_FLUX
    for (int icomp = 0; icomp < 5; ++icomp)
    {
      cell_flux_(index,cell_flux_index_(i,0),icomp) = -iflux[icomp]+vflux[icomp];
    }
#endif

  }
};

#endif
