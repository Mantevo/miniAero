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
#include "Parallel3DMesh.h"

#if WITH_MPI
#include <mpi.h>
#endif


#include <sstream>


Parallel3DMesh::Parallel3DMesh(int nx, int ny, int nz, double lx, double ly, double lz, int problem_type, double angle):
    global_nx_(nx),
    global_ny_(ny),
    global_nz_(nz),
    lx_(lx),
    ly_(ly),
    lz_(lz),
    problem_type_(problem_type),
    ramp_angle(angle)
{


#if WITH_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id_);
#else
  num_procs_= 1;
  my_id_ = 0;
#endif

  //Per processor output
  std::stringstream fs;
  fs << "setupmesh.";
  fs << my_id_;
  std::string filename = fs.str();
  myfile.open(filename.c_str());

  compute_processor_arrangement();

  myfile << "dim: " << nx_ << ", " << ny_ << ", " << nz << std::endl;

  std::vector<int> ghost_dir(3);
  getGhostsDim(ghost_dir);
  myfile << "ghost dim: " << ghost_dir[0] << ", " << ghost_dir[1] << ", " << ghost_dir[2] << std::endl;
  std::vector<int> ghost_offset(3);
  getGhostOffset(ghost_offset);
  myfile << "ghost offset: " << ghost_offset[0] << ", " << ghost_offset[1] << ", " << ghost_offset[2] << std::endl;
}

Parallel3DMesh::~Parallel3DMesh(){
  myfile.close();

}

void Parallel3DMesh::getElementNodeConnectivities(std::vector<int> & element_node_conn, std::vector<int> & elem_global_ids){
    element_node_conn.clear();

    for(int i=0; i<nx_; ++i)
        for(int j=0; j<ny_; ++j)
            for(int k=0; k<nz_; ++k){
                element_node_conn.push_back(getNodeLocalIndex(i,j,k));
                element_node_conn.push_back(getNodeLocalIndex(i+1,j,k));
                element_node_conn.push_back(getNodeLocalIndex(i+1,j+1,k));
                element_node_conn.push_back(getNodeLocalIndex(i,j+1,k));
                element_node_conn.push_back(getNodeLocalIndex(i,j,k+1));
                element_node_conn.push_back(getNodeLocalIndex(i+1,j,k+1));
                element_node_conn.push_back(getNodeLocalIndex(i+1,j+1,k+1));
                element_node_conn.push_back(getNodeLocalIndex(i,j+1,k+1));
                elem_global_ids.push_back(getElementGlobalId(i,j,k));
            }
}

void Parallel3DMesh::getGhostedElementNodeConnectivities(std::vector<int> & element_node_conn, std::vector<int> & elem_global_ids){
    std::vector<int> ghosted_x_index;
    if(nprocx_!=1){
      if(x_block_== 0)
        ghosted_x_index.push_back(nx_);
      else if (x_block_ == nprocx_-1)
        ghosted_x_index.push_back(-1);
      else{
        ghosted_x_index.push_back(-1);
        ghosted_x_index.push_back(nx_);
      }
    }
    for(std::vector<int>::iterator i = ghosted_x_index.begin(); i != ghosted_x_index.end(); ++i)
      for(int j=0; j<ny_; ++j)
        for(int k=0; k<nz_; ++k){
          element_node_conn.push_back(getNodeLocalIndex(*i,j,k));
          element_node_conn.push_back(getNodeLocalIndex(*i+1,j,k));
          element_node_conn.push_back(getNodeLocalIndex(*i+1,j+1,k));
          element_node_conn.push_back(getNodeLocalIndex(*i,j+1,k));
          element_node_conn.push_back(getNodeLocalIndex(*i,j,k+1));
          element_node_conn.push_back(getNodeLocalIndex(*i+1,j,k+1));
          element_node_conn.push_back(getNodeLocalIndex(*i+1,j+1,k+1));
          element_node_conn.push_back(getNodeLocalIndex(*i,j+1,k+1));
          elem_global_ids.push_back(getElementGlobalId(*i,j,k));
        }

    std::vector<int> ghosted_y_index;
    if(nprocy_!=1){
      if(y_block_== 0)
        ghosted_y_index.push_back(ny_);
      else if (y_block_ == nprocy_-1)
        ghosted_y_index.push_back(-1);
      else{
        ghosted_y_index.push_back(-1);
        ghosted_y_index.push_back(ny_);
      }
    }
    for(int i=0; i<nx_; ++i)
      for(std::vector<int>::iterator j = ghosted_y_index.begin(); j != ghosted_y_index.end(); ++j)
        for(int k=0; k<nz_; ++k){
          element_node_conn.push_back(getNodeLocalIndex(i,*j,k));
          element_node_conn.push_back(getNodeLocalIndex(i+1,*j,k));
          element_node_conn.push_back(getNodeLocalIndex(i+1,*j+1,k));
          element_node_conn.push_back(getNodeLocalIndex(i,*j+1,k));
          element_node_conn.push_back(getNodeLocalIndex(i,*j,k+1));
          element_node_conn.push_back(getNodeLocalIndex(i+1,*j,k+1));
          element_node_conn.push_back(getNodeLocalIndex(i+1,*j+1,k+1));
          element_node_conn.push_back(getNodeLocalIndex(i,*j+1,k+1));
          elem_global_ids.push_back(getElementGlobalId(i,*j,k));
        }

    std::vector<int> ghosted_z_index;
    if(nprocz_!=1){
      if(z_block_== 0)
        ghosted_z_index.push_back(nz_);
      else if (z_block_ == nprocz_-1)
        ghosted_z_index.push_back(-1);
      else{
        ghosted_z_index.push_back(-1);
        ghosted_z_index.push_back(nz_);
      }
    }

      for(int i=0; i<nx_; ++i)
        for(int j=0; j<ny_; ++j)
          for(std::vector<int>::iterator k = ghosted_z_index.begin(); k != ghosted_z_index.end(); ++k){
                element_node_conn.push_back(getNodeLocalIndex(i,j,*k));
                element_node_conn.push_back(getNodeLocalIndex(i+1,j,*k));
                element_node_conn.push_back(getNodeLocalIndex(i+1,j+1,*k));
                element_node_conn.push_back(getNodeLocalIndex(i,j+1,*k));
                element_node_conn.push_back(getNodeLocalIndex(i,j,*k+1));
                element_node_conn.push_back(getNodeLocalIndex(i+1,j,*k+1));
                element_node_conn.push_back(getNodeLocalIndex(i+1,j+1,*k+1));
                element_node_conn.push_back(getNodeLocalIndex(i,j+1,*k+1));
                elem_global_ids.push_back(getElementGlobalId(i,j,*k));
            }
}



void Parallel3DMesh::getNodeCoordinates(std::vector<double> & node_coordinates){

  //Note: This includes ghosted nodes which are only used to generate faces which will
  //later be removed.
    node_coordinates.clear();
    std::vector<int> min(3), max(3);
    getNodeLimits(min, max);

    int total_node_size=0;

    for(int i=min[0]; i<max[0]; ++i)
        for(int j=min[1]; j<max[1]; ++j)
            for(int k=min[2]; k<max[2]; ++k){
                std::vector<double> coord = getNodeCoordinate(i, j, k);
                node_coordinates.push_back(coord[0]);
                node_coordinates.push_back(coord[1]);
                node_coordinates.push_back(coord[2]);
                total_node_size=total_node_size+3;
            }
}

void Parallel3DMesh::getTopBCNodes(std::set<int> & bc_nodes){
    int j = ny_;
    for(int i=0; i<nx_+1; ++i)
            for(int k=0; k<nz_+1; ++k){
               bc_nodes.insert(getNodeLocalIndex(i,j,k));
            }
}

void Parallel3DMesh::getBottomBCNodes(std::set<int> & bc_nodes){
    int j = 0;
    for(int i=0; i<nx_+1; ++i)
            for(int k=0; k<nz_+1; ++k){
               bc_nodes.insert(getNodeLocalIndex(i,j,k));
            }
}

void Parallel3DMesh::getRightBCNodes(std::set<int> & bc_nodes){
    int i = nx_;
    for(int j=0; j<ny_+1; ++j)
            for(int k=0; k<nz_+1; ++k){
               bc_nodes.insert(getNodeLocalIndex(i,j,k));
            }
}

void Parallel3DMesh::getLeftBCNodes(std::set<int> & bc_nodes){
    int i = 0;
    for(int j=0; j<ny_+1; ++j)
            for(int k=0; k<nz_+1; ++k){
               bc_nodes.insert(getNodeLocalIndex(i,j,k));
            }
}

void Parallel3DMesh::getBackBCNodes(std::set<int> & bc_nodes){
    int k = nz_;
    for(int i=0; i<nx_+1; ++i)
            for(int j=0; j<ny_+1; ++j){
               bc_nodes.insert(getNodeLocalIndex(i,j,k));
            }
}

void Parallel3DMesh::getFrontBCNodes(std::set<int> & bc_nodes){
    int k = 0;
    for(int i=0; i<nx_+1; ++i)
            for(int j=0; j<ny_+1; ++j){
               bc_nodes.insert(getNodeLocalIndex(i,j,k));
            }
}

void Parallel3DMesh::compute_processor_arrangement(){


  //Check if number of processors is power of 2
  int procs_left = num_procs_;
  int nx_temp = global_nx_;
  int ny_temp = global_ny_;
  int nz_temp = global_nz_;

  nprocx_ = 1;
  nprocy_ = 1;
  nprocz_ = 1;

  int max_size;
  while(procs_left!=1){
    if(procs_left % 2 !=0){
      std::cout << "MPI number of ranks must be a power of 2." << std::endl;
      exit(1);
    }
    procs_left = procs_left/2;

    max_size=nx_temp; if (ny_temp > max_size) max_size=ny_temp; if(nz_temp > max_size) max_size=nz_temp;

    if(nx_temp==max_size){
     nprocx_ = nprocx_ * 2;
     nx_temp = global_nx_/nprocx_;
     continue;
    }
    if(ny_temp==max_size){
     nprocy_ = nprocy_ * 2;
     ny_temp = global_ny_/nprocy_;
     continue;
    }
    if(nz_temp==max_size){
     nprocz_ = nprocz_ * 2;
     nz_temp = global_nz_/nprocz_;
     continue;
    }
  }
  nx_=nx_temp;
  ny_=ny_temp;
  nz_=nz_temp;
  x_block_ = my_id_ % nprocx_;
  int new_size = my_id_/nprocx_;
  y_block_ = new_size % nprocy_;
  new_size = new_size/nprocy_;
  z_block_ = new_size % nprocz_;

  nx_offset_ = global_nx_/nprocx_*(x_block_);
  ny_offset_ = global_ny_/nprocy_*(y_block_);
  nz_offset_ = global_nz_/nprocz_*(z_block_);


  myfile << "Processor division: " << nprocx_ << " , " << nprocy_ << " , " << nprocz_ << std::endl;
  myfile << "block ids: " << my_id_ << " , " << x_block_ << " , " << y_block_ << " , " << z_block_ << std::endl;
  myfile << "Offsets: "  << nx_offset_ << " , " << ny_offset_ << " , " << nz_offset_ << std::endl;
}

#if WITH_MPI
void Parallel3DMesh::setupCommunication(std::vector<int> & elem_global_ids, int num_ghosted_elements,
    std::vector<std::pair<int, int> > & sendProcIdent,
    std::vector<std::pair<int, int> > & recvProcIdent,
    std::vector<int> & sendCount,
    std::vector<int> & recvCount){

  //Ghosted elements on the processor boundary are globally communicated in order to determine which
  //MPI ranks either own or use the ghost.  This process is not scalable.
  //One option is to use the known structure of the generated mesh but then this approach is no longer general.
  int * num_offproc_ghosts = new int[num_procs_];
  MPI_Allgather( &num_ghosted_elements, 1, MPI_INT, num_offproc_ghosts, 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> ghost_offset(num_procs_,0);
  int total_ghosts=0;
  //Allocate memory for ghosts
  for(int i=0; i<num_procs_; ++i){
    if(i==my_id_) continue;
    ghost_offset[i]=total_ghosts;
    total_ghosts+=num_offproc_ghosts[i];
  }

  MPI_Request * send_requests = new MPI_Request[num_procs_-1];
  MPI_Request * recv_requests = new MPI_Request[num_procs_-1];
  MPI_Status * statuses = new MPI_Status[num_procs_-1];

  int * ghost_element_ids = new int[total_ghosts];
  int ghost_start_index = elem_global_ids.size()-(num_ghosted_elements);

  //Send and receive ghosted elements to/from each processor
  int comm_count=0;
  int tag=15;
  for(int i=0; i<num_procs_; ++i){
    if(i==my_id_) continue;
    MPI_Isend(&elem_global_ids[ghost_start_index], num_ghosted_elements, MPI_INT, i, tag, MPI_COMM_WORLD, &send_requests[comm_count]);
    MPI_Irecv(&ghost_element_ids[ghost_offset[i]], num_offproc_ghosts[i], MPI_INT, i, tag, MPI_COMM_WORLD, &recv_requests[comm_count]);
    comm_count++;
  }
  MPI_Waitall(comm_count, send_requests, statuses);
  MPI_Waitall(comm_count, recv_requests, statuses);

  //Search to determine whether owner of the ghosted element of the other processors.
  //Use a binary search here for efficiency.
  int num_nonghosted_elements=elem_global_ids.size()-num_ghosted_elements;
  std::vector<int>::iterator end_iter = elem_global_ids.begin()+num_nonghosted_elements;
  std::vector<int> sort_global_ids(elem_global_ids.begin(), end_iter );
  std::sort(sort_global_ids.begin(), sort_global_ids.end());
  for(int i = 0; i < num_procs_; ++i){
    if(i==my_id_) continue;
    for(int j=0; j<num_offproc_ghosts[i]; j++){
      std::vector<int>::iterator found_iter;
      int value = ghost_element_ids[ghost_offset[i]+j];
      found_iter = std::lower_bound(sort_global_ids.begin(), sort_global_ids.end(), value);
      if(found_iter!=sort_global_ids.end() && !(value < *found_iter))
        sendProcIdent.push_back(std::make_pair(i,*found_iter));
    }
  }


  //Setup the point-to-point communication now that we who the ghosted elements are owned by.
  //For solution computation, only point-to-point communication is needed.
  sendCount.clear();
  sendCount.resize(num_procs_,0);
  recvCount.clear();
  recvCount.resize(num_procs_,0);

  for(int i=0; i<sendProcIdent.size(); ++i){
    sendCount[sendProcIdent[i].first]+=1;
  }

  comm_count=0;
  tag=15;
  for(int i=0; i<num_procs_; ++i){
    if(i==my_id_) continue;
    //Send to all other procs
    MPI_Isend(&sendCount[i], 1, MPI_INT, i, tag, MPI_COMM_WORLD, &send_requests[comm_count]);
    MPI_Irecv(&recvCount[i], 1, MPI_INT, i, tag, MPI_COMM_WORLD, &recv_requests[comm_count]);
    comm_count++;
  }
  MPI_Waitall(comm_count, send_requests, statuses);
  MPI_Waitall(comm_count, recv_requests, statuses);

  int total_recvCount=0;
  int total_sendCount=0;
  for(int i=0; i<num_procs_; ++i){
    total_recvCount += recvCount[i];
    total_sendCount += sendCount[i];
  }


  std::vector<int> recvIdent(total_recvCount, 0);
  std::vector<int> sendIdent(total_sendCount, 0);
  for(int i=0; i<total_sendCount; ++i){
    sendIdent[i] = sendProcIdent[i].second;
  }

  int send_comm_count=0, recv_comm_count=0;
  tag=25;
  int send_offset=0;
  int recv_offset=0;
  for(int i=0; i<num_procs_; ++i){
    if(i==my_id_) continue;
    if(sendCount[i]!=0){
      MPI_Isend(&sendIdent[send_offset], sendCount[i], MPI_INT, i, tag, MPI_COMM_WORLD, &send_requests[send_comm_count]);
      send_offset+=sendCount[i];
      send_comm_count++;
    }
    if(recvCount[i]!=0){
      MPI_Irecv(&recvIdent[recv_offset], recvCount[i], MPI_INT, i, tag, MPI_COMM_WORLD, &recv_requests[recv_comm_count]);
      recv_offset+=recvCount[i];
      recv_comm_count++;
    }
  }
  MPI_Waitall(send_comm_count, send_requests, statuses);
  MPI_Waitall(recv_comm_count, recv_requests, statuses);
  int index=0;
  for(int i=0; i<num_procs_; ++i)
    for(int j=0; j<recvCount[i]; ++j){
      recvProcIdent.push_back(std::make_pair(i,recvIdent[index]));
      index++;
    }

  delete [] send_requests;
  delete [] recv_requests;
  delete [] statuses;
  delete [] ghost_element_ids;
}
#endif
