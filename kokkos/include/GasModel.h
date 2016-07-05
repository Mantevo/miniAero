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
#ifndef INCLUDE_GASMODEL_H_
#define INCLUDE_GASMODEL_H_
#include <cmath>

/*Functions to compute gas properties from the conserved variables
 * or primitive variables using an ideal gas model.
 */

   template<class DeviceType >
   KOKKOS_INLINE_FUNCTION double ComputePressure(const double* V)
   {
      const double Rgas = 287.05;
      const double rho = V[0];
      const double T = V[4];

      return rho*Rgas*T;
   }

   // ------------------------------------------------------------------------------------------------

   template<class DeviceType >
   KOKKOS_INLINE_FUNCTION double ComputeSoundSpeed(const double* V)
   {
      const double gamma = 1.4;
      const double rho = V[0];

      const double pressure = ComputePressure<DeviceType>(V);

      return std::sqrt(gamma * pressure / rho);
   }

   // ------------------------------------------------------------------------------------------------

   template<class DeviceType >
   KOKKOS_INLINE_FUNCTION double ComputeEnthalpy(const double* V)
   {
      const double Cp = 1004.0;
      const double T = V[4];
      return Cp*T;
   }

   template<class DeviceType >
   KOKKOS_INLINE_FUNCTION void ComputePrimitives(const double* U, double* V)
   {
       double gamma = 1.4;
       double Rgas = 287.05;
       double r, u, v, w, T, ri, k, e;

       r  = U[0];
       ri = 1.0 / r;
       u  = U[1] * ri;
       v  = U[2] * ri;
       w  = U[3] * ri;
       k  = 0.5 * (u * u + v * v + w * w);
       e  = U[4] * ri - k;
       T  = e * (gamma - 1.0) / Rgas;

       V[0] = r;
       V[1] = u;
       V[2] = v;
       V[3] = w;
       V[4] = T;
   }

   template<class DeviceType >
   KOKKOS_INLINE_FUNCTION double ComputeViscosity(const double temperature) 
   {
      const double sutherland_0 = 1.458e-6;
      const double sutherland_1 = 110.4;
      return sutherland_0 * temperature * std::sqrt(temperature) / (temperature + sutherland_1);
   }

   template<class DeviceType >
   KOKKOS_INLINE_FUNCTION double ComputeThermalConductivity(const double viscosity) 
   {
      const double Pr = 0.71;
      const double Cp = 1006.0;
      return viscosity*Cp/Pr; 
   }


#endif
