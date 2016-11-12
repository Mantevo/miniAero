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
 * CopyGhost.C
 *
 *  Created on: Apr 9, 2014
 *      Author: kjfrank
 */

#include "CopyGhost.h"
#if WITH_MPI
#include <mpi.h>
#endif

void communicate_ghosted_cell_data(std::vector<int> & sendCount, std::vector<int> & recvCount,
    double *send_data, double *recv_data, int data_per_cell)
{
#ifdef WITH_MPI
  int num_procs, my_id;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  // communicate values to other processors
  MPI_Request * send_requests = new MPI_Request[num_procs-1];
  MPI_Request * recv_requests = new MPI_Request[num_procs-1];
  MPI_Status * statuses = new MPI_Status[num_procs-1];
  int send_comm_count=0, recv_comm_count=0;
  int tag=35;
  int send_offset=0;
  int recv_offset=0;
  for(int i=0; i<num_procs; ++i){
    if(i==my_id) continue;
    if(sendCount[i]!=0){
      int data_length = sendCount[i]*data_per_cell;
      MPI_Isend(&send_data[send_offset], data_length, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &send_requests[send_comm_count]);
      send_offset+=data_length;
      send_comm_count++;
    }
    if(recvCount[i]!=0){
      int data_length = recvCount[i]*data_per_cell;
      MPI_Irecv(&recv_data[recv_offset], data_length, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &recv_requests[recv_comm_count]);
      recv_offset+=data_length;
      recv_comm_count++;
    }
  }
  MPI_Waitall(send_comm_count, send_requests, statuses);
  MPI_Waitall(recv_comm_count, recv_requests, statuses);

  delete [] send_requests;
  delete [] recv_requests;
  delete [] statuses;

#endif
}




