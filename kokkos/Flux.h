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
#ifndef _INCLUDE_FLUX_H_
#define _INCLUDE_FLUX_H_

#include <cstdio>
#include <Kokkos_Core.hpp>
#include "MathToolsDevice.h"
#include "Faces.h"
#include "VanAlbadaLimiter.h"
#include "GasModel.h"

#ifdef ATOMICS_FLUX
#include "Kokkos_Atomic.hpp"
#endif

/* compute_face_flux
 * functor to compute the internal face flux contributions.
 * Uses the templated Inviscid and Inviscid flux types to
 * compute the contribution. This functor organizes
 * the data to pass to the functions that compute the flux
 * and puts the flux contribution in the appropriate place
 * using either Gather-Sum or Atomics for thread safety.
 */

template<class Device, bool second_order, class InviscidFluxType,
    class ViscousFluxType>
struct compute_face_flux {
  typedef Device device_type;
  typedef typename ViewTypes<Device>::c_rnd_solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::c_rnd_face_cell_conn_type face_cell_conn_type;
  typedef typename ViewTypes<Device>::cell_storage_field_type cell_storage_field_type;
  typedef typename ViewTypes<Device>::c_vector_field_type vector_field_type;
  typedef typename ViewTypes<Device>::c_rnd_gradient_field_type gradient_field_type;

  face_cell_conn_type face_cell_conn_;
  face_cell_conn_type cell_flux_index_;
  solution_field_type cell_values_;
  gradient_field_type cell_gradients_;
  solution_field_type cell_limiters_;
  vector_field_type cell_coordinates_;
  cell_storage_field_type cell_flux_;
  vector_field_type face_coordinates_, face_normal_, face_tangent_,
      face_binormal_;
  Kokkos::View<const int*, Device> permute_vector_;
  InviscidFluxType inviscid_flux_evaluator_;
  ViscousFluxType viscous_flux_evaluator_;


  compute_face_flux(Faces<Device> faces, solution_field_type cell_values,
      gradient_field_type cell_gradients, solution_field_type cell_limiters,
      Cells<Device> cells, InviscidFluxType inviscid_flux,
      ViscousFluxType viscous_flux) :
      face_cell_conn_(faces.face_cell_conn_), cell_flux_index_(
          faces.cell_flux_index_), cell_values_(cell_values), cell_gradients_(
          cell_gradients), cell_limiters_(cell_limiters), cell_coordinates_(
          cells.coordinates_), cell_flux_(cells.cell_flux_), face_coordinates_(
          faces.coordinates_), face_normal_(faces.face_normal_), face_tangent_(
          faces.face_tangent_), face_binormal_(faces.face_binormal_), inviscid_flux_evaluator_(
          inviscid_flux), viscous_flux_evaluator_(viscous_flux) , permute_vector_(faces.permute_vector_) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& ii) const {
    const int i = permute_vector_(ii);
    const int left_index = face_cell_conn_(i, 0);
    const int right_index = face_cell_conn_(i, 1);

    double flux[5];
    double conservatives_l[5];
    double conservatives_r[5];
    double primitives_l[5];
    double primitives_r[5];

    for (int icomp = 0; icomp < 5; ++icomp) {
      conservatives_l[icomp] = cell_values_(left_index, icomp);
      conservatives_r[icomp] = cell_values_(right_index, icomp);
    }

    ComputePrimitives<device_type>(conservatives_l, primitives_l);
    ComputePrimitives<device_type>(conservatives_r, primitives_r);

    if (second_order) {

      //Extrapolation
      for (int icomp = 0; icomp < 5; ++icomp) {
      	double gradient_primitive_l_tmp = 0;
	      double gradient_primitive_r_tmp = 0;

        for (int idir = 0; idir < 3; ++idir) {
    	    gradient_primitive_l_tmp += (face_coordinates_(i, idir)
            	- cell_coordinates_(left_index, idir))
		          * cell_gradients_(left_index, icomp, idir);

    	    gradient_primitive_r_tmp += (face_coordinates_(i, idir)
		          - cell_coordinates_(right_index, idir))
		          * cell_gradients_(right_index, icomp, idir);
        }

        primitives_l[icomp] += gradient_primitive_l_tmp *
                cell_limiters_(left_index, icomp);
        primitives_r[icomp] += gradient_primitive_r_tmp *
                cell_limiters_(right_index, icomp);
      }

    } // End of second order


    inviscid_flux_evaluator_.compute_flux(primitives_l, primitives_r, flux,
        &face_normal_(i,0), &face_tangent_(i,0), &face_binormal_(i,0));

    if (ViscousFluxType::isViscous) {
      double primitives_face[5];
      double gradients_face[5][3];

      for (int icomp = 0; icomp < 5; ++icomp) {
        primitives_face[icomp] = 0.5
            * (primitives_l[icomp] + primitives_r[icomp]);

        for (int idir = 0; idir < 3; ++idir) {
          gradients_face[icomp][idir] = 0.5
              * (cell_gradients_(left_index, icomp, idir)
                  + cell_gradients_(right_index, icomp, idir));
        }
      }

      double vflux[5];
      viscous_flux_evaluator_.compute_flux(gradients_face, primitives_face,
          &face_normal_(i,0), vflux);

      for (int icomp = 0; icomp < 5; ++icomp) {
        flux[icomp] -= vflux[icomp];
      }
    }

#ifdef ATOMICS_FLUX
    for (int icomp = 0; icomp < 5; ++icomp)
    {
      double * left_cell = &cell_flux_(left_index,0,icomp);
      Kokkos::atomic_add(left_cell, -flux[icomp]);
      double * right_cell = &cell_flux_(right_index,0,icomp);
      Kokkos::atomic_add(right_cell, flux[icomp]);
    }
#endif

#ifdef CELL_FLUX
    for (int icomp = 0; icomp < 5; ++icomp)
    {
      cell_flux_(left_index,cell_flux_index_(i,0),icomp) = -flux[icomp];
      cell_flux_(right_index,cell_flux_index_(i,1),icomp) = flux[icomp];
    }
#endif

  }

};

/* compute_face_flux
 * functor add the flux contributions to the residual
 * uses either gather-sum or atomics for thread safety
 */
template<class Device>
struct apply_cell_flux {

  typedef Device device_type;
  typedef typename ViewTypes<Device>::scalar_field_type scalar_field_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;
  typedef typename ViewTypes<Device>::cell_storage_field_type cell_storage_field_type;
  typedef typename ViewTypes<Device>::cell_face_conn_type cell_face_conn_type;

  int number_faces_;
  scalar_field_type volume_;
  cell_storage_field_type flux_;
  solution_field_type residuals_;

  double dt_;
  double vol_;

  apply_cell_flux(Cells<Device> cells, solution_field_type residuals, double dt) :
      number_faces_(cells.nfaces_), volume_(cells.volumes_), flux_(
          cells.cell_flux_), residuals_(residuals), dt_(dt) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {

    for (int icomp = 0; icomp < 5; ++icomp) {
      residuals_(i, icomp) = 0.0;
    }
#ifdef ATOMICS_FLUX
    for(int flux_id=0; flux_id<1; ++flux_id)
#else
    for (int flux_id = 0; flux_id < number_faces_; ++flux_id)
#endif

        {
      for (int icomp = 0; icomp < 5; ++icomp) {
        residuals_(i, icomp) = residuals_(i, icomp)
            + dt_ / volume_(i) * flux_(i, flux_id, icomp);
      }
    }
  }
};

#endif
