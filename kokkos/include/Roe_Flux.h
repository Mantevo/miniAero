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
#ifndef INCLUDE_ROE_FLUX_H_
#define INCLUDE_ROE_FLUX_H_

#include <Kokkos_View.hpp>
#include <cmath>
#include "GasModel.h"
#include "MathToolsDevice.h"
#include "Faces.h"

/*roe_flux
 * Functor to compute the Roe flux based on two states.
 * See Roe, Journal of Computational Physics, Volume 43, Issue 2, October 1981, Pages 357–372.
 */
template<class Device>
struct roe_flux {
  typedef Device device_type;
  roe_flux() {
  }

  KOKKOS_INLINE_FUNCTION
  void compute_flux(const double * primitives_l, const double * primitives_r,
      double * flux, double * a_vec, double * t_vec, double * a_x_t) const {

    //Eigenvalue fix constants.
    const double efix_u = 0.1;
    const double efix_c = 0.1;

    const double gm1 = 0.4;

    // Left state
    const double rl = primitives_l[0];
    const double ul = primitives_l[1];
    const double vl = primitives_l[2];
    const double wl = primitives_l[3];

    const double pl = ComputePressure<device_type>(primitives_l);
    const double Cl = ComputeSoundSpeed<device_type>(primitives_l);
    const double hl = ComputeEnthalpy<device_type>(primitives_l);

    const double kel = 0.5 * (ul * ul + vl * vl + wl * wl);
    const double htl = hl + kel;
    const double ubl = a_vec[0] * ul + a_vec[1] * vl + a_vec[2] * wl;

    // Right state
    const double rr = primitives_r[0];
    const double ur = primitives_r[1];
    const double vr = primitives_r[2];
    const double wr = primitives_r[3];

    const double pr = ComputePressure<device_type>(primitives_r);
    const double Cr = ComputeSoundSpeed<device_type>(primitives_r);
    const double hr = ComputeEnthalpy<device_type>(primitives_r);

    const double ker = 0.5 * (ur * ur + vr * vr + wr * wr);
    const double htr = hr + ker;
    const double ubr = a_vec[0] * ur + a_vec[1] * vr + a_vec[2] * wr;

    const double mdotl = rl * ubl;
    const double mdotr = rr * ubr;

    const double pl_plus_pr = pl + pr;

    // Central part
    flux[0] = 0.5 * (mdotl + mdotr);
    flux[1] = 0.5 * (mdotl * ul + mdotr * ur + a_vec[0] * pl_plus_pr);
    flux[2] = 0.5 * (mdotl * vl + mdotr * vr + a_vec[1] * pl_plus_pr);
    flux[3] = 0.5 * (mdotl * wl + mdotr * wr + a_vec[2] * pl_plus_pr);
    flux[4] = 0.5 * (mdotl * htl + mdotr * htr);

    // Upwinded part
    const double a_vec_norm = MathTools<device_type>::Vec3Norm(a_vec);
    const double t_vec_norm = MathTools<device_type>::Vec3Norm(t_vec);
    const double a_x_t_norm = MathTools<device_type>::Vec3Norm(a_x_t);

    const double a_vec_unit[] = { a_vec[0] / a_vec_norm, a_vec[1] / a_vec_norm,
        a_vec[2] / a_vec_norm };
    const double t_vec_unit[] = { t_vec[0] / t_vec_norm, t_vec[1] / t_vec_norm,
        t_vec[2] / t_vec_norm };
    const double a_x_t_unit[] = { a_x_t[0] / a_x_t_norm, a_x_t[1] / a_x_t_norm,
        a_x_t[2] / a_x_t_norm };

    const double denom = 1.0 / (std::sqrt(rl) + std::sqrt(rr));
    const double alpha = sqrt(rl) * denom;
    const double beta = 1.0 - alpha;

    const double ua = alpha * ul + beta * ur;
    const double va = alpha * vl + beta * vr;
    const double wa = alpha * wl + beta * wr;
    const double ha = alpha * hl + beta * hr
        + 0.5 * alpha * beta
            * (std::pow(ur - ul, 2) + std::pow(vr - vl, 2)
                + std::pow(wr - wl, 2));
    const double Ca = std::sqrt(gm1 * ha);

    // Compute flux matrices
    double roe_mat_l[25];
    double roe_mat_r[25];

    const double ub = ua * a_vec_unit[0] + va * a_vec_unit[1]
        + wa * a_vec_unit[2];
    const double vb = ua * t_vec_unit[0] + va * t_vec_unit[1]
        + wa * t_vec_unit[2];
    const double wb = ua * a_x_t_unit[0] + va * a_x_t_unit[1]
        + wa * a_x_t_unit[2];
    const double keb = 0.5 * (ua * ua + va * va + wa * wa);
    const double c2i = 1.0 / (Ca * Ca);
    const double hc2 = 0.5 * c2i;

    // Left matrix
    roe_mat_l[0] = gm1 * (keb - ha) + Ca * (Ca - ub);
    roe_mat_l[1] = Ca * a_vec_unit[0] - gm1 * ua;
    roe_mat_l[2] = Ca * a_vec_unit[1] - gm1 * va;
    roe_mat_l[3] = Ca * a_vec_unit[2] - gm1 * wa;
    roe_mat_l[4] = gm1;

    roe_mat_l[5] = gm1 * (keb - ha) + Ca * (Ca + ub);
    roe_mat_l[6] = -Ca * a_vec_unit[0] - gm1 * ua;
    roe_mat_l[7] = -Ca * a_vec_unit[1] - gm1 * va;
    roe_mat_l[8] = -Ca * a_vec_unit[2] - gm1 * wa;
    roe_mat_l[9] = gm1;

    roe_mat_l[10] = keb - ha;
    roe_mat_l[11] = -ua;
    roe_mat_l[12] = -va;
    roe_mat_l[13] = -wa;
    roe_mat_l[14] = 1.0;

    roe_mat_l[15] = -vb;
    roe_mat_l[16] = t_vec_unit[0];
    roe_mat_l[17] = t_vec_unit[1];
    roe_mat_l[18] = t_vec_unit[2];
    roe_mat_l[19] = 0.0;

    roe_mat_l[20] = -wb;
    roe_mat_l[21] = a_x_t_unit[0];
    roe_mat_l[22] = a_x_t_unit[1];
    roe_mat_l[23] = a_x_t_unit[2];
    roe_mat_l[24] = 0.0;

    // Right matrix
    roe_mat_r[0] = hc2;
    roe_mat_r[1] = hc2;
    roe_mat_r[2] = -gm1 * c2i;
    roe_mat_r[3] = 0.0;
    roe_mat_r[4] = 0.0;

    roe_mat_r[5] = (ua + a_vec_unit[0] * Ca) * hc2;
    roe_mat_r[6] = (ua - a_vec_unit[0] * Ca) * hc2;
    roe_mat_r[7] = -gm1 * ua * c2i;
    roe_mat_r[8] = t_vec_unit[0];
    roe_mat_r[9] = a_x_t_unit[0];

    roe_mat_r[10] = (va + a_vec_unit[1] * Ca) * hc2;
    roe_mat_r[11] = (va - a_vec_unit[1] * Ca) * hc2;
    roe_mat_r[12] = -gm1 * va * c2i;
    roe_mat_r[13] = t_vec_unit[1];
    roe_mat_r[14] = a_x_t_unit[1];

    roe_mat_r[15] = (wa + a_vec_unit[2] * Ca) * hc2;
    roe_mat_r[16] = (wa - a_vec_unit[2] * Ca) * hc2;
    roe_mat_r[17] = -gm1 * wa * c2i;
    roe_mat_r[18] = t_vec_unit[2];
    roe_mat_r[19] = a_x_t_unit[2];

    roe_mat_r[20] = (ha + keb + Ca * ub) * hc2;
    roe_mat_r[21] = (ha + keb - Ca * ub) * hc2;
    roe_mat_r[22] = (Ca * Ca - gm1 * (ha + keb)) * c2i;
    roe_mat_r[23] = vb;
    roe_mat_r[24] = wb;

    // Conservative variable jumps
    double U_jmp[] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    U_jmp[0] = rr - rl;
    U_jmp[1] = rr * ur - rl * ul;
    U_jmp[2] = rr * vr - rl * vl;
    U_jmp[3] = rr * wr - rl * wl;
    U_jmp[4] = (rr * htr - pr) - (rl * htl - pl);

    // Compute CFL number
    const double cbar = Ca * a_vec_norm;
    const double ubar = ua * a_vec[0] + va * a_vec[1] + wa * a_vec[2];
    const double cfl = std::abs(ubar) + cbar;

    // Eigenvalue fix
    const double eig1 = ubar + cbar;
    const double eig2 = ubar - cbar;
    const double eig3 = ubar;

    double abs_eig1 = std::abs(eig1);
    double abs_eig2 = std::abs(eig2);
    double abs_eig3 = std::abs(eig3);

    const double epuc = efix_u * cfl;
    const double epcc = efix_c * cfl;

    // Original Roe eigenvalue fix
    if (abs_eig1 < epcc) abs_eig1 = 0.5 * (eig1 * eig1 + epcc * epcc) / epcc;
    if (abs_eig2 < epcc) abs_eig2 = 0.5 * (eig2 * eig2 + epcc * epcc) / epcc;
    if (abs_eig3 < epuc) abs_eig3 = 0.5 * (eig3 * eig3 + epuc * epuc) / epuc;

    double eigp[] = { 0.5 * (eig1 + abs_eig1), 0.5 * (eig2 + abs_eig2), 0.5
        * (eig3 + abs_eig3), 0.0, 0.0 };
    eigp[3] = eigp[4] = eigp[2];

    double eigm[] = { 0.5 * (eig1 - abs_eig1), 0.5 * (eig2 - abs_eig2), 0.5
        * (eig3 - abs_eig3), 0.0, 0.0 };
    eigm[3] = eigm[4] = eigm[2];

    // Compute upwind flux
    double ldq[] = { 0, 0, 0, 0, 0 };
    double lldq[] = { 0, 0, 0, 0, 0 };
    double rlldq[] = { 0, 0, 0, 0, 0 };

    MathTools<device_type>::MatVec(5, 1.0, roe_mat_l, U_jmp, 0.0, ldq);

    for (int j = 0; j < 5; ++j)
      lldq[j] = (eigp[j] - eigm[j]) * ldq[j];

    MathTools<device_type>::MatVec(5, 1.0, roe_mat_r, lldq, 0.0, rlldq);

    for (int icomp = 0; icomp < 5; ++icomp)
      flux[icomp] -= 0.5*rlldq[icomp];

  }
};

#endif
