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
#ifndef INCLUDE_VISCOUS_FLUX_H_
#define INCLUDE_VISCOUS_FLUX_H_

#include <Kokkos_View.hpp>
#include "GasModel.h"

/* no_viscous_flux
 * functor that computes no viscous flux for inviscid calculation
 * Uses the enum isViscous=false to remove unneeded code at compile time.
 */
template <class Device>
struct no_viscous_flux {
  enum {isViscous = false};
  typedef Device device_type;
  no_viscous_flux(){}
  KOKKOS_INLINE_FUNCTION
  void compute_flux(double grad_primitive[5][3], double * primitive, 
     double * a_vec, double * vflux) const
  {
    return;
  }
};

/* newtonian_viscous_flux
 * functor that computer Newtonian viscous flux using gradients
 * and primitive values.
 */
template <class Device>
struct newtonian_viscous_flux {
  enum {isViscous = true};
  typedef Device device_type;
  newtonian_viscous_flux(){}
  KOKKOS_INLINE_FUNCTION
  void compute_flux(double grad_primitive[5][3], double * primitive,
     double * a_vec, double * vflux) const
  {
  double viscosity = ComputeViscosity<device_type>(primitive[4]);
  double thermal_conductivity = ComputeThermalConductivity<device_type>(viscosity);
  double divergence_velocity = 0;

  for(int icomp=0; icomp<5; ++icomp){
    vflux[icomp] = 0.0;
  }

  for(int i=0; i<3; i++){
    divergence_velocity += grad_primitive[i+1][i];
  }

  for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        const double delta_ij = (i == j ) ? 1 : 0;
        const double S_ij = 0.5*(grad_primitive[i+1][j] + grad_primitive[j+1][i]);
        const double t_ij = S_ij - divergence_velocity*delta_ij/3.;
        vflux[1+i] += (2*viscosity*t_ij)*a_vec[j];
        vflux[4] += (2*viscosity*t_ij)*primitive[i+1]*a_vec[j];
      }
    vflux[4] += thermal_conductivity*grad_primitive[4][i]*a_vec[i];
    }


  return;
  }
};

#endif
