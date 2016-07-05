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
#ifndef INCLUDE_MATH_TOOLS_H_
#define INCLUDE_MATH_TOOLS_H_

#include <cmath>
#include <Kokkos_View.hpp>
#include <Kokkos_Cuda.hpp>


/*MathTools
 * Class that contains some convenience math functions that are
 * templated to run on device.
 */
template<class DeviceType >
class MathTools
{

public:

   // ----------------------------------------------------------------------------
   KOKKOS_INLINE_FUNCTION static double Vec3Norm(const double a[])
   {
      return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
   }

   KOKKOS_INLINE_FUNCTION static void MatVec(const int N, const double alpha, const double A[], const double x[], const double beta, double y[])
   {
      for(int i=0; i < N; ++i) {
        y[i] *= beta;
        for(int j=0; j < N; ++j) {
          y[i] += alpha * A[N*i + j] * x[j];
        }
      }
   }

   KOKKOS_INLINE_FUNCTION static double min(double val1, double val2)
   {
     return std::min(val1, val2);
   }
   KOKKOS_INLINE_FUNCTION static double max(double val1, double val2)
   {
     return std::max(val1, val2);
   }
};

template <>
KOKKOS_INLINE_FUNCTION double MathTools<Kokkos::Cuda>::min(double val1, double val2)
  {
    return fmin(val1, val2);
  } 

template <>
KOKKOS_INLINE_FUNCTION double MathTools<Kokkos::Cuda>::max(double val1, double val2)
  {
    return fmax(val1, val2);
  } 


#endif
