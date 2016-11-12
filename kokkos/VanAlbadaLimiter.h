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
#ifndef _INCLUDE_VANALBADALIMITER_H_
#define _INCLUDE_VANALBADALIMITER_H_

#include <Kokkos_Core.hpp>
#include <float.h>
#include "MathToolsDevice.h"

/* VanAlbadaLimiter
 * Class with single function that computes the the VanAlbada limiter value.
 */
template<class DeviceType >
class VanAlbadaLimiter
{

public:

 KOKKOS_INLINE_FUNCTION static double limit(double dumax, double dumin, double du){
 double yval = 2;

  if (du > DBL_EPSILON)
  {
    yval = dumax/du;
  }
  else if (du < -DBL_EPSILON)
  {
    yval = dumin/du;
  }

  double phi = 1;
  if (yval < 2){
    phi = (4*yval-yval*yval)/(yval*yval-4*yval+8);
    phi = MathTools<DeviceType>::max(phi,0.0);
    phi = MathTools<DeviceType>::min(phi,1.0);
  }

  return phi;
  }
};

#endif
