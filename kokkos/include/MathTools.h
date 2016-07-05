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
#include <algorithm>



/*MathTools
 * Class that contains some convenience math functions.
 */

class MathTools
{
private:

protected:

public:

   static void Vec3Cross(const double a[], const double b[], double c[])
   {
      c[0] = a[1] * b[2] - b[1] * a[2];
      c[1] = -a[0] * b[2] + b[0] * a[2];
      c[2] = a[0] * b[1] - b[0] * a[1];
   }


   static void Vec3Average(const double a[], const double b[], double c[])
   {
      c[0] = 0.5 * (a[0] + b[0]);
      c[1] = 0.5 * (a[1] + b[1]);
      c[2] = 0.5 * (a[2] + b[2]);
   }


   static void Vec3SumInto(const double a[], double c[])
   {
      c[0] += a[0];
      c[1] += a[1];
      c[2] += a[2];
   }

   static void Vec3Scale(const double scalar, double c[])
   {
      c[0] *= scalar;
      c[1] *= scalar;
      c[2] *= scalar;
   }

   static double VecNDot(const unsigned N, const double a[], const double b[])
   {
      double result = 0;
      for(unsigned i=0; i < N; ++i)
        result += a[i] * b[i];
      return result;
   }


};

#endif
