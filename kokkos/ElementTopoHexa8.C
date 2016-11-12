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
#if WITH_MPI
#include <mpi.h>
#endif
#include "ElementTopoHexa8.h"

#include <algorithm>
#include <cstdio>
#include <cassert>
#include <iostream>

#include "MathTools.h"


// Static variable declarations
// Gauss point info
const double* ElementTopoHexa8::xi_(NULL);
const double* ElementTopoHexa8::eta_(NULL);
const double* ElementTopoHexa8::zeta_(NULL);

ElementTopoHexa8::ElementTopoHexa8()
{
  num_gp_ = 8;
  xi_ = hex_xi_2x2x2;
  eta_ = hex_eta_2x2x2;
  zeta_ = hex_zeta_2x2x2;
  wts_ = hex_wts_2x2x2;
}

ElementTopoHexa8::~ElementTopoHexa8()
{

}

int ElementTopoHexa8::Eval_detJ(const unsigned& gauss_pt, const double* ex, const double* ey,
      const double* ez, double& detJ, double* J, double* dNdxi, double* dNdeta, double* dNdzeta,
      double* dummy) const
{
   this->Eval_dNdn(gauss_pt, dNdxi, dNdeta, dNdzeta, dummy);

   J[0] = MathTools::VecNDot(8, dNdxi, ex);
   J[1] = MathTools::VecNDot(8, dNdxi, ey);
   J[2] = MathTools::VecNDot(8, dNdxi, ez);

   J[3] = MathTools::VecNDot(8, dNdeta, ex);
   J[4] = MathTools::VecNDot(8, dNdeta, ey);
   J[5] = MathTools::VecNDot(8, dNdeta, ez);

   J[6] = MathTools::VecNDot(8, dNdzeta, ex);
   J[7] = MathTools::VecNDot(8, dNdzeta, ey);
   J[8] = MathTools::VecNDot(8, dNdzeta, ez);

   detJ = J[0] * (J[4] * J[8] - J[5] * J[7])
        + J[1] * (J[5] * J[6] - J[3] * J[8])
        + J[2] * (J[3] * J[7] - J[4] * J[6]);

   if (detJ < DETJ_EPS)
   {
     int my_id;
#if WITH_MPI
     MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
#else
     my_id=0;
#endif
      std::cout << "Warning in ElementTopoHexa8::Eval_detJ: Jacobian determinant < 0 on processor:" << my_id << std::endl;
      return -1;
   }

   return 0;
}

void ElementTopoHexa8::Eval_dNdn(const unsigned& gauss_pt, double* dNdxi, double* dNdeta, double* dNdzeta,
      double* dummy) const
{
   const double xi = xi_[gauss_pt];
   const double eta = eta_[gauss_pt];
   const double zeta = zeta_[gauss_pt];

   dNdxi[0] = -0.125 * (1.0 - eta) * (1.0 - zeta);
   dNdxi[1] = 0.125 * (1.0 - eta) * (1.0 - zeta);
   dNdxi[2] = 0.125 * (1.0 + eta) * (1.0 - zeta);
   dNdxi[3] = -0.125 * (1.0 + eta) * (1.0 - zeta);
   dNdxi[4] = -0.125 * (1.0 - eta) * (1.0 + zeta);
   dNdxi[5] = 0.125 * (1.0 - eta) * (1.0 + zeta);
   dNdxi[6] = 0.125 * (1.0 + eta) * (1.0 + zeta);
   dNdxi[7] = -0.125 * (1.0 + eta) * (1.0 + zeta);

   dNdeta[0] = -0.125 * (1.0 - xi) * (1.0 - zeta);
   dNdeta[1] = -0.125 * (1.0 + xi) * (1.0 - zeta);
   dNdeta[2] = 0.125 * (1.0 + xi) * (1.0 - zeta);
   dNdeta[3] = 0.125 * (1.0 - xi) * (1.0 - zeta);
   dNdeta[4] = -0.125 * (1.0 - xi) * (1.0 + zeta);
   dNdeta[5] = -0.125 * (1.0 + xi) * (1.0 + zeta);
   dNdeta[6] = 0.125 * (1.0 + xi) * (1.0 + zeta);
   dNdeta[7] = 0.125 * (1.0 - xi) * (1.0 + zeta);

   dNdzeta[0] = -0.125 * (1.0 - xi) * (1.0 - eta);
   dNdzeta[1] = -0.125 * (1.0 + xi) * (1.0 - eta);
   dNdzeta[2] = -0.125 * (1.0 + xi) * (1.0 + eta);
   dNdzeta[3] = -0.125 * (1.0 - xi) * (1.0 + eta);
   dNdzeta[4] = 0.125 * (1.0 - xi) * (1.0 - eta);
   dNdzeta[5] = 0.125 * (1.0 + xi) * (1.0 - eta);
   dNdzeta[6] = 0.125 * (1.0 + xi) * (1.0 + eta);
   dNdzeta[7] = 0.125 * (1.0 - xi) * (1.0 + eta);
}

double ElementTopoHexa8::ComputeVolume(const double* ex, const double* ey, const double* ez) const
{
   double volume = 0.0, detJ = 0;

   double  J[9], dNdxi[8], dNdeta[8], dNdzeta[8];

   for (unsigned ig = 0; ig < 8; ++ig)
   {
      this->Eval_detJ(ig, ex, ey, ez, detJ, J, dNdxi, dNdeta, dNdzeta, NULL);

      volume += detJ;
   }

   return volume;
}
