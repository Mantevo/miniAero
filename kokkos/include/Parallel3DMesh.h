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
#ifndef INCLUDE_PARALLEL_3D_MESH_H_
#define INCLUDE_PARALLEL_3D_MESH_H_
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <cmath>

#include <Kokkos_View.hpp>
#include "MeshData.h"
#include "Face.h"
#include "Cell.h"
#include "MeshProcessor.h"

/*Parallel3DMesh
 * Class that manages setup of an in-code generated mesh.
 * Generates meshes that are structured but using unstructured data structures.
 * The number of processors must be powers of 2.
 * For perfect load-balancing number of points in each direction must be a power of 2.
 * Ghosting information is also determined by this class.
 * This class could potentially be replaced by a read-in from an unstructured mesh database file.
 */

class Parallel3DMesh{
public:
    Parallel3DMesh(int nx, int ny, int nz, double lx, double ly, double lz, int problem_type, double angle = 0.0);
    ~Parallel3DMesh();

    void getGhostsDim(std::vector<int> & ghost_dir){

      ghost_dir[0] = 0;
      ghost_dir[1] = 0;
      ghost_dir[2] = 0;


      if(nprocx_!=1)
        if(x_block_==0 || x_block_ == nprocx_-1)
          ghost_dir[0] = ghost_dir[0]+1;
        else
          ghost_dir[0] = ghost_dir[0]+2;
      if(nprocy_!=1)
        if(y_block_==0 || y_block_ == nprocy_-1)
          ghost_dir[1] = ghost_dir[1]+1;
        else
          ghost_dir[1] = ghost_dir[1]+2;
      if(nprocz_!=1)
        if(z_block_==0 || z_block_ == nprocz_-1)
          ghost_dir[2] = ghost_dir[2]+1;
        else
          ghost_dir[2] = ghost_dir[2]+2;
    }

    void getGhostOffset(std::vector<int> & ghost_offset){

      ghost_offset[0] = 0;
      ghost_offset[1] = 0;
      ghost_offset[2] = 0;

      if(nprocx_!=1 && x_block_!=0)
        ghost_offset[0]=1;
      if(nprocy_!=1 && y_block_!=0)
        ghost_offset[1]=1;
      if(nprocz_!=1 && z_block_!=0)
        ghost_offset[2]=1;
    }

    void getNodeLimits(std::vector<int> & min, std::vector<int> & max){
      min[0] = 0, max[0] = nx_+1;
      min[1] = 0, max[1] = ny_+1;
      min[2] = 0, max[2] = nz_+1;

      if(nprocx_!=1){
        if(x_block_== 0)
          max[0]=nx_+2;
        else if (x_block_ ==nprocx_-1)
          min[0]=-1;
        else{
          min[0]=-1;
          max[0]=nx_+2;
        }
      }
      if(nprocy_!=1){
        if(y_block_== 0)
          max[1]=ny_+2;
        else if (y_block_ ==nprocy_-1)
          min[1]=-1;
        else{
          min[1]=-1;
          max[1]=ny_+2;
        }
      }
      if(nprocz_!=1){
        if(z_block_== 0)
          max[2]=nz_+2;
        else if (z_block_ ==nprocz_-1)
          min[2]=-1;
        else{
          min[2]=-1;
          max[2]=nz_+2;
        }
      }
    }

    inline int getNumberGhostedElements(){
      std::vector<int> ghost_dir(3);
      getGhostsDim(ghost_dir);

      return ghost_dir[0]*ny_*nz_+ghost_dir[1]*nx_*nz_+ghost_dir[2]*nx_*ny_;
    }

    inline int getNumberElements(){
      return nx_*ny_*nz_+getNumberGhostedElements();
    }

    inline int getNumberNodes(){

      std::vector<int> ghost_dir(3);
      getGhostsDim(ghost_dir);

      return (nx_+ghost_dir[0]+1)*(ny_+ghost_dir[1]+1)*(nz_+ghost_dir[2]+1);
    }



    inline int getNumberGhostedNodes(){

      std::vector<int> ghost_dir(3);
      getGhostsDim(ghost_dir);

      return ghost_dir[0]*(ny_+1)*(nz_+1)+ghost_dir[1]*(nx_+1)*(nz_+1)+ghost_dir[2]*(nx_+1)*(ny_+1);
    }



    template <class Device>
    void fillMeshData(struct MeshData<Device> & mesh_data){

    int num_nodes = getNumberNodes();
    int num_elems = getNumberElements();
    int num_ghosted_elems = getNumberGhostedElements();

    myfile << "Total Number of Nodes: " << num_nodes << std::endl;
    myfile << "Total Number of Elements: " << num_elems << std::endl;
    myfile << "Number of Ghosted Elements: " << num_ghosted_elems << std::endl;

    std::vector<int> element_node_conn, element_global_id;

    int num_connectivities = num_elems*8;
    element_node_conn.reserve(num_connectivities);
    element_global_id.reserve(num_elems);
    getElementNodeConnectivities(element_node_conn, element_global_id);
    getGhostedElementNodeConnectivities(element_node_conn, element_global_id);

    int node_coord_size = num_nodes*3;
    std::vector<double> node_coordinates;
    node_coordinates.reserve(node_coord_size);
    getNodeCoordinates(node_coordinates);

    std::vector<Face> mesh_faces;
    std::vector<Cell> mesh_cells(num_elems);

    time_t faceStartTime=0, faceEndTime=0;
    time(&faceStartTime);
    create_faces(element_node_conn, mesh_faces, node_coordinates, num_elems, num_nodes);
    time(&faceEndTime);
    double faceElapsedTime = difftime(faceEndTime,faceStartTime);
    if(my_id_==0){
      fprintf(stdout,"\n ... Face creation time: %8.2f seconds ...\n", faceElapsedTime);
    }

    compute_cell_volumes(mesh_cells, element_node_conn, node_coordinates, num_elems, num_ghosted_elems);
    myfile << "Done computing cell volumes" << std::endl;
    compute_cell_centroid(mesh_cells, element_node_conn, node_coordinates, num_elems);
    myfile << "Done compute cell centroids" << std::endl;

    delete_ghosted_faces(mesh_faces,num_elems-num_ghosted_elems);
    //Faces - Interior and BC
    std::vector<Face> bc_faces;
    extract_BC_faces(mesh_faces, bc_faces);

    faceStartTime=0, faceEndTime=0;
    time(&faceStartTime);
    myfile << "BC Faces num:" << bc_faces.size() << std::endl;

    //Nodes on top of domain
    std::set<int> top_nodes;
    getTopBCNodes(top_nodes);
    std::vector<Face> top_faces;
    organize_BC_faces(bc_faces, top_faces, top_nodes);
    myfile << "Top_faces num: " << top_faces.size() << std::endl;

    //Nodes on bottom of domain
    std::set<int> bottom_nodes;
    getBottomBCNodes(bottom_nodes);
    std::vector<Face> bottom_faces;
    organize_BC_faces(bc_faces, bottom_faces, bottom_nodes);
    myfile << "bottom_faces num: " << bottom_faces.size() << std::endl;

    //Nodes on right of domain
    std::set<int> right_nodes;
    getRightBCNodes(right_nodes);
    std::vector<Face> right_faces;
    organize_BC_faces(bc_faces, right_faces, right_nodes);
    myfile << "right_faces num: " << right_faces.size() << std::endl;

    //Nodes on left of domain
    std::set<int> left_nodes;
    getLeftBCNodes(left_nodes);
    std::vector<Face> left_faces;
    organize_BC_faces(bc_faces, left_faces, left_nodes);
    myfile << "left_faces num: " << left_faces.size() << std::endl;

    //Nodes on front of domain
    std::set<int> front_nodes;
    getFrontBCNodes(front_nodes);
    std::vector<Face> front_faces;
    organize_BC_faces(bc_faces, front_faces, front_nodes);
    myfile << "front_faces num: " << front_faces.size() << std::endl;

    //Nodes on back of domain
    std::set<int> back_nodes;
    getBackBCNodes(back_nodes);
    std::vector<Face> back_faces;
    organize_BC_faces(bc_faces, back_faces, back_nodes);
    myfile << "back_faces num: " << back_faces.size() << std::endl;

    time(&faceEndTime);
    faceElapsedTime = difftime(faceEndTime,faceStartTime);
    if(my_id_==0){
      fprintf(stdout,"\n ... Extract BC face and delete ghost time: %8.2f seconds ...\n", faceElapsedTime);
    }


    //Parallel communication setup
    #if WITH_MPI
      if(my_id_==0)
        std::cout << "Start setup communication." << std::endl;
      std::vector<std::pair<int, int> > sendProcIdent;
      std::vector<std::pair<int, int> > recvProcIdent;

      time_t commStartTime=0, commEndTime=0;
      time(&commStartTime);
      setupCommunication(element_global_id, num_ghosted_elems, sendProcIdent, recvProcIdent, mesh_data.sendCount, mesh_data.recvCount);
      time(&commEndTime);
      double commElapsedTime = difftime(commEndTime,commStartTime);
      if(my_id_==0){
        fprintf(stdout,"\n ... Setup Communcation function time: %8.2f seconds ...\n", commElapsedTime);
      }

      commStartTime=0, commEndTime=0;
      time(&commStartTime);
      std::sort(sendProcIdent.begin(),sendProcIdent.end());
      std::sort(recvProcIdent.begin(),recvProcIdent.end());

      std::vector<int> & sendProcessorOffset = mesh_data.send_offset;
      std::vector<int> & recvProcessorOffset = mesh_data.recv_offset;
      sendProcessorOffset.resize(num_procs_,0);
      recvProcessorOffset.resize(num_procs_,0);

      for(int i=1; i<num_procs_; ++i){
          sendProcessorOffset[i]+=sendProcessorOffset[i-1]+mesh_data.sendCount[i-1];
          recvProcessorOffset[i]+=recvProcessorOffset[i-1]+mesh_data.recvCount[i-1];
      }

      std::vector<int> sendLocalID, recvLocalID;
      for(int i=0; i<sendProcIdent.size();++i)
      {
        for(int j=0; j<element_global_id.size(); ++j)
          if(element_global_id[j]==sendProcIdent[i].second){
            sendLocalID.push_back(j);
            break;
          }
      }
      for(int i=0; i<recvProcIdent.size();++i)
      {
        for(int j=0; j<element_global_id.size(); ++j)
          if(element_global_id[j]==recvProcIdent[i].second){
            recvLocalID.push_back(j);
            break;
          }
      }

      time(&commEndTime);
      commElapsedTime = difftime(commEndTime,commStartTime);
      if(my_id_==0){
        fprintf(stdout,"\n ... Rest of setup communication time: %8.2f seconds ...\n", commElapsedTime);
      }
      if(my_id_==0)
        std::cout << "End setup communication." << std::endl;
    #endif


    //Fill Kokkos Arrays on HostMirror and copy to Device.

    //Internal Faces
    int ninternal_faces  = mesh_faces.size();
    myfile << "internal face size: " << ninternal_faces << std::endl;

    Faces<Device> internal_faces(ninternal_faces, 2);

    //Boundary Faces
    int ntop_faces = top_faces.size();
    int nbottom_faces = bottom_faces.size();
    int nright_faces = right_faces.size();
    int nleft_faces = left_faces.size();
    int nfront_faces = front_faces.size();
    int nback_faces = back_faces.size();
    Faces<Device> top_boundary_faces(ntop_faces, 1);
    Faces<Device> bottom_boundary_faces(nbottom_faces, 1);
    Faces<Device> right_boundary_faces(nright_faces, 1);
    Faces<Device> left_boundary_faces(nleft_faces, 1);
    Faces<Device> front_boundary_faces(nfront_faces, 1);
    Faces<Device> back_boundary_faces(nback_faces, 1);


    //Shuffle internal faces
    std::random_shuffle(mesh_faces.begin(), mesh_faces.end());

    //Fill Faces on Device (through HostMirror)
    copy_faces(internal_faces, mesh_faces);
    copy_faces(top_boundary_faces, top_faces);
    copy_faces(bottom_boundary_faces, bottom_faces);
    copy_faces(right_boundary_faces, right_faces);
    copy_faces(left_boundary_faces, left_faces);
    copy_faces(front_boundary_faces, front_faces);
    copy_faces(back_boundary_faces, back_faces);


    //Cells
    const int faces_per_element = 6;
    Cells<Device> device_cells(num_elems, faces_per_element);//ncells, faces/element. - Need centroid?
    copy_cell_data(device_cells, mesh_cells);

    //Fill Cells on device.
    mesh_data.mesh_cells = device_cells;
    mesh_data.internal_faces = internal_faces;
    if(problem_type_ == 1)
      mesh_data.boundary_faces.push_back(std::pair<std::string, Faces<Device> >("NoSlip", bottom_boundary_faces));
    else
      mesh_data.boundary_faces.push_back(std::pair<std::string, Faces<Device> >("Tangent", bottom_boundary_faces));
    if(problem_type_ == 1)
      mesh_data.boundary_faces.push_back(std::pair<std::string, Faces<Device> >("Extrapolate", top_boundary_faces));
    else
      mesh_data.boundary_faces.push_back(std::pair<std::string, Faces<Device> >("Tangent", top_boundary_faces));
    mesh_data.boundary_faces.push_back(std::pair<std::string, Faces<Device> >("Tangent", front_boundary_faces));
    mesh_data.boundary_faces.push_back(std::pair<std::string, Faces<Device> >("Tangent", back_boundary_faces));
    mesh_data.boundary_faces.push_back(std::pair<std::string, Faces<Device> >("Extrapolate", right_boundary_faces));
    if(problem_type_ == 0)
      mesh_data.boundary_faces.push_back(std::pair<std::string, Faces<Device> >("Extrapolate", left_boundary_faces));
    else
      mesh_data.boundary_faces.push_back(std::pair<std::string, Faces<Device> >("Inflow", left_boundary_faces));


    #ifdef WITH_MPI
      //Communication data on device
      typedef typename Kokkos::View<int *, Device> id_map_type;
      typedef typename Kokkos::View<int *, Device>::HostMirror id_map_type_host;
      id_map_type send_local_ids("send_local_ids", sendProcIdent.size());
      id_map_type recv_local_ids("recv_local_ids", recvProcIdent.size());
      id_map_type_host send_local_ids_host = Kokkos::create_mirror(send_local_ids);
      id_map_type_host recv_local_ids_host = Kokkos::create_mirror(recv_local_ids);

      for(int i=0; i<sendLocalID.size(); ++i)
        send_local_ids_host(i)=sendLocalID[i];
      for(int i=0; i<recvLocalID.size(); ++i)
        recv_local_ids_host(i)=recvLocalID[i];

      Kokkos::deep_copy(send_local_ids, send_local_ids_host);
      Kokkos::deep_copy(recv_local_ids, recv_local_ids_host);
      mesh_data.send_local_ids = send_local_ids;
      mesh_data.recv_local_ids = recv_local_ids;
    #endif
    mesh_data.num_ghosts = num_ghosted_elems;
    mesh_data.num_owned_cells = num_elems-num_ghosted_elems;


    }

private:

    int getNodeLocalIndex(int i, int j, int k){
      std::vector<int> ghost_offset(3);
      getGhostOffset(ghost_offset);
      std::vector<int> ghost_dir(3);
      getGhostsDim(ghost_dir);

      return (i+ghost_offset[0])*((ny_+1+ghost_dir[1])*(nz_+1+ghost_dir[2]))+(j+ghost_offset[1])*(nz_+1+ghost_dir[2])+(k+ghost_offset[2]);
    }

    inline int getElementGlobalId(int i, int j, int k){
      return (nx_offset_+i)*(global_ny_*global_nz_)+(ny_offset_+j)*global_nz_+(nz_offset_+k);
    }

    std::vector<double> getNodeCoordinate(int i, int j, int k){

      int global_i = nx_offset_+i;
      int global_j = ny_offset_+j;
      int global_k = nz_offset_+k;


      const double PI = 3.14159265;
      std::vector<double> coordinate(3);
      coordinate[0]= (double)global_i/(global_nx_)*lx_;
      coordinate[2]= (double)global_k/(global_nz_)*lz_;

      if(coordinate[0] < lx_/2.0){
        coordinate[1]= (double)global_j/(global_ny_)*ly_;
      }
      else{
        double y_ramp = (coordinate[0]-(lx_/2.0))*tan(ramp_angle*PI/180.0);
        double ly_scaled = ly_-y_ramp;
        coordinate[1] = y_ramp+(double)global_j*ly_scaled/(global_ny_);
      }
      return coordinate;
    }

    void getElementNodeConnectivities(std::vector<int> & element_node_conn, std::vector<int> & elem_global_ids);
    void getGhostedElementNodeConnectivities(std::vector<int> & element_node_conn, std::vector<int> & elem_global_ids);
    void getNodeCoordinates(std::vector<double> & node_coordinates);
    void getGhostedNodeCoordinates(std::vector<double> & node_coordinates);

    void getTopBCNodes(std::set<int> & bc_nodes);
    void getBottomBCNodes(std::set<int> & bc_nodes);
    void getRightBCNodes(std::set<int> & bc_nodes);
    void getLeftBCNodes(std::set<int> & bc_nodes);
    void getFrontBCNodes(std::set<int> & bc_nodes);
    void getBackBCNodes(std::set<int> & bc_nodes);

    void compute_processor_arrangement();

    void setupCommunication(std::vector<int> & elem_global_ids, int num_ghosted_elements,
        std::vector<std::pair<int, int> > & sendProcIdent,
        std::vector<std::pair<int, int> > & recvProcIdent,
        std::vector<int> & sendCount,
        std::vector<int> & recvCount);

    int num_procs_, my_id_;
    int x_block_, y_block_, z_block_;
    int nprocx_, nprocy_, nprocz_;
    int global_nx_, global_ny_, global_nz_;
    int nx_, ny_, nz_;
    int nx_offset_, ny_offset_, nz_offset_;
    double lx_, ly_, lz_;
    int problem_type_;
    double ramp_angle;

    std::ofstream myfile;
};



#endif
