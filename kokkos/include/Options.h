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
/*
 * Options.h
 *
 *  Created on: Jan 15, 2014
 *      Author: kjfrank
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_

#include <fstream>
#include <iostream>

/*Options
 * struct containing options that are read-in from the input file.
 * The input file is always named miniaero.inp
 */

struct Options{

  int problem_type;
  double lx, ly, lz, angle;
  int nx, ny, nz;
  int ntimesteps;
  double dt;
  int nthreads;
  int output_results;
  int output_frequency;
  int second_order_space;
  int viscous;

  Options():
    problem_type(0),
    nx(10),
    ny(10),
    nz(10),
    ntimesteps(1),
    dt(5e-8),
    nthreads(1),
    output_results(0),
    output_frequency(10),
    second_order_space(0)
  {

  }

  void read_options_file(){
    //Options file should look like:
    // problem_type (0 - Sod, 1 - Viscous Flat Plate, 2 - Inviscid Ramp)
    // lx ly lz ramp_angle (either SOD(angle=0)  or ramp problem)
    // nx ny nz
    // ntimesteps
    // dt
    // nthreads
    // output_results (0 - no, anything else yes)
    // information output_frequency
    // second order space (0 - no, anything else yes)
    // viscous (0 - no, anything else yes)

    std::ifstream option_file( "miniaero.inp" );

    if(option_file){}
    else {std::cout << "miniaero.inp does not exist." << std::endl;}

    option_file >> problem_type;
    option_file >> lx >> ly >> lz >> angle;
    option_file >> nx >> ny >> nz;
    option_file >> ntimesteps;
    option_file >> dt;
    option_file >> nthreads;
    option_file >> output_results;
    option_file >> output_frequency;
    option_file >> second_order_space;
    option_file >> viscous;
    option_file.close();
  }
};






#endif /* OPTIONS_H_ */
