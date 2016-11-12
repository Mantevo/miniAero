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
 *WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICuvel_leftAR PURPOSE ARE DISCLAIMED.
 *IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 *LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*/
#ifndef INCLUDE_ROE_FLUX_H_
#define INCLUDE_ROE_FLUX_H_

#include <Kokkos_Core.hpp>
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
    typedef Device execution_space;
  
    roe_flux() {
    }

  KOKKOS_INLINE_FUNCTION
  void compute_flux(const double * const& primitives_left, const double * const& primitives_right,
                    double * const& flux,
                    const double * const& face_normal,
                    const double * const& face_tangent,
                    const double * const& face_binormal) const {

    //Eigenvalue fix constants.
    const double efix_u = 0.1;
    const double efix_c = 0.1;

    const double gm1 = 0.4;

    // Left state
    const double rho_left  = primitives_left[0];
    const double uvel_left = primitives_left[1];
    const double vvel_left = primitives_left[2];
    const double wvel_left = primitives_left[3];

    const double pressure_left = ComputePressure<execution_space>(primitives_left);
    const double enthalpy_left = ComputeEnthalpy<execution_space>(primitives_left);

    const double total_enthalpy_left = enthalpy_left +0.5 * (uvel_left * uvel_left + vvel_left * vvel_left + wvel_left * wvel_left);
    const double mass_flux_left = rho_left * (face_normal[0] * uvel_left +
                                              face_normal[1] * vvel_left +
                                              face_normal[2] * wvel_left);
    
    // Right state
    const double rho_right = primitives_right[0];
    const double uvel_right = primitives_right[1];
    const double vvel_right = primitives_right[2];
    const double wvel_right = primitives_right[3];
      
    const double pressure_right = ComputePressure<execution_space>(primitives_right);
    const double enthalpy_right = ComputeEnthalpy<execution_space>(primitives_right);

    const double total_enthalpy_right = enthalpy_right + 0.5 * (uvel_right * uvel_right + vvel_right * vvel_right + wvel_right * wvel_right);
    const double mass_flux_right = rho_right *(face_normal[0] * uvel_right +
                                                 face_normal[1] * vvel_right +
                                                 face_normal[2] * wvel_right);

    const double pressure_sum = pressure_left + pressure_right;

    // Central flux contribution part
    flux[0] = 0.5 * (mass_flux_left + mass_flux_right);
    flux[1] = 0.5 * (mass_flux_left * uvel_left + mass_flux_right * uvel_right + face_normal[0] * pressure_sum);
    flux[2] = 0.5 * (mass_flux_left * vvel_left + mass_flux_right * vvel_right + face_normal[1] * pressure_sum);
    flux[3] = 0.5 * (mass_flux_left * wvel_left + mass_flux_right * wvel_right + face_normal[2] * pressure_sum);
    flux[4] = 0.5 * (mass_flux_left * total_enthalpy_left + mass_flux_right * total_enthalpy_right);

    // Upwinded part
    const double face_normal_norm = std::sqrt(face_normal[0] * face_normal[0] +
        face_normal[1] * face_normal[1] + face_normal[2] * face_normal[2]);
      
    //const double face_normal_norm = MathTools<execution_space>::Vec3Norm(face_normal);
    const double face_tangent_norm = std::sqrt(face_tangent[0] * face_tangent[0] +
                                                 face_tangent[1] * face_tangent[1] +
                                                 face_tangent[2] * face_tangent[2]);

    //const double face_binormal_norm = MathTools<execution_space>::Vec3Norm(face_binormal);
    const double face_binormal_norm = std::sqrt(face_binormal[0] * face_binormal[0] +
                                                face_binormal[1] * face_binormal[1] +
                                                face_binormal[2] * face_binormal[2]);

    const double face_normal_unit[] = { face_normal[0] / face_normal_norm, face_normal[1] / face_normal_norm,
        face_normal[2] / face_normal_norm };
    const double face_tangent_unit[] = { face_tangent[0] / face_tangent_norm, face_tangent[1] / face_tangent_norm,
        face_tangent[2] / face_tangent_norm };
    const double face_binormal_unit[] = { face_binormal[0] / face_binormal_norm, face_binormal[1] / face_binormal_norm,
        face_binormal[2] / face_binormal_norm };

    const double denom = 1.0 / (std::sqrt(rho_left) + std::sqrt(rho_right));
    const double alpha = sqrt(rho_left) * denom;
    const double beta = 1.0 - alpha;

    const double uvel_roe = alpha * uvel_left + beta * uvel_right;
    const double vvel_roe = alpha * vvel_left + beta * vvel_right;
    const double wvel_roe = alpha * wvel_left + beta * wvel_right;
    const double enthalpy_roe = alpha * enthalpy_left + beta * enthalpy_right +
                0.5 * alpha * beta *
                ((uvel_right - uvel_left) * (uvel_right - uvel_left) +
                (vvel_right - vvel_left) * (vvel_right - vvel_left) +
                (wvel_right - wvel_left) * (wvel_right - wvel_left));
    const double speed_sound_roe = std::sqrt(gm1 * enthalpy_roe);

    // Compute flux matrices
    double roe_mat_eigenvectors[25];
    //double roe_mat_right_eigenvectors[25];

    const double normal_velocity = uvel_roe * face_normal_unit[0] + vvel_roe * face_normal_unit[1]
        + wvel_roe * face_normal_unit[2];
    const double tangent_velocity = uvel_roe * face_tangent_unit[0] + vvel_roe * face_tangent_unit[1]
        + wvel_roe * face_tangent_unit[2];
    const double binormal_velocity = uvel_roe * face_binormal_unit[0] + vvel_roe * face_binormal_unit[1]
        + wvel_roe * face_binormal_unit[2];
    const double kinetic_energy_roe = 0.5 * (uvel_roe * uvel_roe + vvel_roe * vvel_roe + wvel_roe * wvel_roe);
    const double speed_sound_squared_inverse = 1.0 / (speed_sound_roe * speed_sound_roe);
    const double half_speed_sound_squared_inverse = 0.5 * speed_sound_squared_inverse;

    // Conservative variable jumps
    double conserved_jump[] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    conserved_jump[0] = rho_right - rho_left;
    conserved_jump[1] = rho_right * uvel_right - rho_left * uvel_left;
    conserved_jump[2] = rho_right * vvel_right - rho_left * vvel_left;
    conserved_jump[3] = rho_right * wvel_right - rho_left * wvel_left;
    conserved_jump[4] = (rho_right * total_enthalpy_right - pressure_right) - (rho_left * total_enthalpy_left - pressure_left);

    // Compute CFL number
    const double cbar = speed_sound_roe * face_normal_norm;
    const double ubar = uvel_roe * face_normal[0] +
            vvel_roe * face_normal[1] +
            wvel_roe * face_normal[2];
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
    //double lldq[] = { 0, 0, 0, 0, 0 };
    double rlldq[] = { 0, 0, 0, 0, 0 };
      
    // Left matrix
    roe_mat_eigenvectors[0] = gm1 * (kinetic_energy_roe - enthalpy_roe) + speed_sound_roe * (speed_sound_roe - normal_velocity);
    roe_mat_eigenvectors[1] = speed_sound_roe * face_normal_unit[0] - gm1 * uvel_roe;
    roe_mat_eigenvectors[2] = speed_sound_roe * face_normal_unit[1] - gm1 * vvel_roe;
    roe_mat_eigenvectors[3] = speed_sound_roe * face_normal_unit[2] - gm1 * wvel_roe;
    roe_mat_eigenvectors[4] = gm1;
      
    roe_mat_eigenvectors[5] = gm1 * (kinetic_energy_roe - enthalpy_roe) + speed_sound_roe * (speed_sound_roe + normal_velocity);
    roe_mat_eigenvectors[6] = -speed_sound_roe * face_normal_unit[0] - gm1 * uvel_roe;
    roe_mat_eigenvectors[7] = -speed_sound_roe * face_normal_unit[1] - gm1 * vvel_roe;
    roe_mat_eigenvectors[8] = -speed_sound_roe * face_normal_unit[2] - gm1 * wvel_roe;
    roe_mat_eigenvectors[9] = gm1;
      
    roe_mat_eigenvectors[10] = kinetic_energy_roe - enthalpy_roe;
    roe_mat_eigenvectors[11] = -uvel_roe;
    roe_mat_eigenvectors[12] = -vvel_roe;
    roe_mat_eigenvectors[13] = -wvel_roe;
    roe_mat_eigenvectors[14] = 1.0;
      
    roe_mat_eigenvectors[15] = -tangent_velocity;
    roe_mat_eigenvectors[16] = face_tangent_unit[0];
    roe_mat_eigenvectors[17] = face_tangent_unit[1];
    roe_mat_eigenvectors[18] = face_tangent_unit[2];
    roe_mat_eigenvectors[19] = 0.0;
      
    roe_mat_eigenvectors[20] = -binormal_velocity;
    roe_mat_eigenvectors[21] = face_binormal_unit[0];
    roe_mat_eigenvectors[22] = face_binormal_unit[1];
    roe_mat_eigenvectors[23] = face_binormal_unit[2];
    roe_mat_eigenvectors[24] = 0.0;

    MathTools<execution_space>::MatVec5(1.0, roe_mat_eigenvectors, conserved_jump, 0.0, ldq);

    for (int j = 0; j < 5; ++j)
      ldq[j] = (eigp[j] - eigm[j]) * ldq[j];
      
    // Right matrix
    roe_mat_eigenvectors[0] = half_speed_sound_squared_inverse;
    roe_mat_eigenvectors[1] = half_speed_sound_squared_inverse;
    roe_mat_eigenvectors[2] = -gm1 * speed_sound_squared_inverse;
    roe_mat_eigenvectors[3] = 0.0;
    roe_mat_eigenvectors[4] = 0.0;
      
    roe_mat_eigenvectors[5] = (uvel_roe + face_normal_unit[0] * speed_sound_roe) * half_speed_sound_squared_inverse;
    roe_mat_eigenvectors[6] = (uvel_roe - face_normal_unit[0] * speed_sound_roe) * half_speed_sound_squared_inverse;
    roe_mat_eigenvectors[7] = -gm1 * uvel_roe * speed_sound_squared_inverse;
    roe_mat_eigenvectors[8] = face_tangent_unit[0];
    roe_mat_eigenvectors[9] = face_binormal_unit[0];
      
    roe_mat_eigenvectors[10] = (vvel_roe + face_normal_unit[1] * speed_sound_roe) * half_speed_sound_squared_inverse;
    roe_mat_eigenvectors[11] = (vvel_roe - face_normal_unit[1] * speed_sound_roe) * half_speed_sound_squared_inverse;
    roe_mat_eigenvectors[12] = -gm1 * vvel_roe * speed_sound_squared_inverse;
    roe_mat_eigenvectors[13] = face_tangent_unit[1];
    roe_mat_eigenvectors[14] = face_binormal_unit[1];
      
    roe_mat_eigenvectors[15] = (wvel_roe + face_normal_unit[2] * speed_sound_roe) * half_speed_sound_squared_inverse;
    roe_mat_eigenvectors[16] = (wvel_roe - face_normal_unit[2] * speed_sound_roe) * half_speed_sound_squared_inverse;
    roe_mat_eigenvectors[17] = -gm1 * wvel_roe * speed_sound_squared_inverse;
    roe_mat_eigenvectors[18] = face_tangent_unit[2];
    roe_mat_eigenvectors[19] = face_binormal_unit[2];
      
    roe_mat_eigenvectors[20] = (enthalpy_roe + kinetic_energy_roe + speed_sound_roe * normal_velocity) * half_speed_sound_squared_inverse;
    roe_mat_eigenvectors[21] = (enthalpy_roe + kinetic_energy_roe - speed_sound_roe * normal_velocity) * half_speed_sound_squared_inverse;
    roe_mat_eigenvectors[22] = (speed_sound_roe * speed_sound_roe - gm1 * (enthalpy_roe + kinetic_energy_roe)) * speed_sound_squared_inverse;
    roe_mat_eigenvectors[23] = tangent_velocity;
    roe_mat_eigenvectors[24] = binormal_velocity;

    MathTools<execution_space>::MatVec5(1.0, roe_mat_eigenvectors, ldq, 0.0, rlldq);

    for (int icomp = 0; icomp < 5; ++icomp)
      flux[icomp] -= 0.5*rlldq[icomp];

  }
};

#endif
