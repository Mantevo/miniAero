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
#ifndef INCLUDE_INITIAL_CONDITIONS_H_
#define INCLUDE_INITIAL_CONDITIONS_H_
#include "Cells.h"

/* initialize_sod3d
 * functor to set initialize conditions
 * sets up the SOD problem (shock-tube problem)
 * which is really 1D but can be run 2D or 3D.
 */
template <class Device>
struct initialize_sod3d{

  typedef Device     device_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;

  struct Cells<Device> cells_;
  solution_field_type soln_,solnp1_;
  double midx_;

  initialize_sod3d(struct Cells<Device> cells, solution_field_type soln, solution_field_type solnp1, double midx) :
  cells_(cells),
  soln_(soln),
  solnp1_(solnp1),
  midx_(midx)
{}

  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{

    const double Rgas = 287.05;
    const double gamma = 1.4;
    const double Cv = Rgas/(gamma-1.0);

    double P1=68947.57;
    double T1=288.889;
    double P2=6894.757;
    double T2=231.11;


    double density1 = P1/(Rgas*T1);
    double rhoE1 = density1*(Cv*T1);

    double density2 = P2/(Rgas*T2);
    double rhoE2 = density2*(Cv*T2);
    double x = cells_.coordinates_(i,0);

    if(x<midx_){
      soln_(i,0)=density1;
      soln_(i,1)=0.0;
      soln_(i,2)=0.0;
      soln_(i,3)=0.0;
      soln_(i,4)=rhoE1;
      for(int icomp=0; icomp<5; icomp++){
        solnp1_(i,icomp)=soln_(i, icomp);
      }
    }
    else{
      soln_(i,0)=density2;
      soln_(i,1)=0.0;
      soln_(i,2)=0.0;
      soln_(i,3)=0.0;
      soln_(i,4)=rhoE2;
      for(int icomp=0; icomp<5; icomp++){
        solnp1_(i,icomp)=soln_(i, icomp);
      }
    }
  }
};

/* initialize_constant
 * functor to set initialize conditions
 * set constant condition for entire flow field
 * This constant state is passed in as flow_state
 */
template <class Device>
struct initialize_constant{

  typedef Device     device_type;
  typedef typename ViewTypes<Device>::solution_field_type solution_field_type;

  struct Cells<Device> cells_;
  solution_field_type soln_,solnp1_;
  double flow_state_[5];

  initialize_constant(struct Cells<Device> cells, solution_field_type soln, solution_field_type solnp1, double * flow_state) :
    cells_(cells),
    soln_(soln),
    solnp1_(solnp1)
  {
    for(int i=0; i<5; ++i)
      flow_state_[i]=flow_state[i];
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( int i )const{
    soln_(i,0)=flow_state_[0];
    soln_(i,1)=flow_state_[1];
    soln_(i,2)=flow_state_[2];
    soln_(i,3)=flow_state_[3];
    soln_(i,4)=flow_state_[4];
    for(int icomp=0; icomp<5; icomp++){
      solnp1_(i,icomp)=soln_(i, icomp);
    }
  }
};





#endif
