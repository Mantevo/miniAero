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
#include "Face.h"
#include "Cell.h"
#include "MeshProcessor.h"
#include "ElementTopoHexa8.h"


#include <algorithm>
#include <list>
#include <iostream>

void create_faces(std::vector<int> & element_node_conn, std::vector<Face> & faces, std::vector<double> & node_coordinates, int num_elems, int num_nodes){
    //FIXME: Hard coded to Hex8 for now.
    const int num_elem_faces = 6;
    const int num_face_nodes = 4;
    const int num_elem_nodes = 8;
    const int face_node_ordering[] = { 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7, 0, 3, 2, 1, 4, 5, 6, 7 };
    int * elem_node_ids;

    int face_node_ids[num_face_nodes];
    // Node-to-face storage
    std::vector < std::vector<int> > node_face_list(num_nodes);
    std::vector<FaceData> facedata;
    int face_num = 0;
    int cell_num = 0;


    // Loop over volume Elements and build CV and Face data structures

    for (int ie = 0; ie < num_elems; ++ie)
    {
        //Get Element nodes
        elem_node_ids = &element_node_conn[ie*num_elem_nodes];
        int elem_face_num = 0;

       // Loop over the Faces of this element
       for (int ifa = 0, pos = 0; ifa < num_elem_faces; ++ifa)
       {
          bool face_has_been_assigned = false;

          const int* face_node_order = face_node_ordering + (pos);

          for (int ifn = 0; ifn < num_face_nodes; ++ifn)
          {
             face_node_ids[ifn] = elem_node_ids[face_node_order[ifn]];
          }
          pos += num_face_nodes;

          const int min_face_node_lid = *std::min_element(face_node_ids, face_node_ids + num_face_nodes);

          std::list<int> face_node_lid_list(face_node_ids, face_node_ids + num_face_nodes);
          face_node_lid_list.sort();

          for (int ifn = 0; ifn < node_face_list[min_face_node_lid].size(); ++ifn)
          {
             const int face_index = node_face_list[min_face_node_lid][ifn];

             std::list<int> face_node_lid_list2(facedata[face_index].node_lids.begin(),
                   facedata[face_index].node_lids.end());
             face_node_lid_list2.sort();

             if (std::equal(face_node_lid_list.begin(), face_node_lid_list.end(), face_node_lid_list2.begin()))
             {
                face_has_been_assigned = true;
                facedata[face_index].cv_plus = cell_num;
                facedata[face_index].flux_index_plus = elem_face_num;
                ++elem_face_num;
                break;
             }
          }

          if (face_has_been_assigned){
             continue;
          }

          node_face_list[min_face_node_lid].push_back(face_num);

          FaceData face;

          face.cv_minus = cell_num;
          face.flux_index_minus = elem_face_num;
          ++elem_face_num;
          for (int ifn = 0; ifn < num_face_nodes; ++ifn)
          {
             face.node_lids.push_back(face_node_ids[ifn]);
          }
          face.min_node_lid = min_face_node_lid;
          facedata.push_back(face);

          ++face_num;
       } // Loop over the Faces of this element

       ++cell_num;
    }
    faces.clear();
    faces.reserve(face_num);
    for(int ifa = 0; ifa < face_num; ++ifa){
        Face new_face(facedata[ifa].cv_minus, facedata[ifa].cv_plus, facedata[ifa].flux_index_minus, facedata[ifa].flux_index_plus, num_face_nodes,facedata[ifa].node_lids, node_coordinates);
        faces.push_back(new_face);
    }
}

void compute_cell_volumes(std::vector<Cell> & cells, std::vector<int> & element_node_conn, std::vector<double> & node_coordinates, int num_elems, int num_ghosted){
    const int num_elem_nodes = 8;
    int * elem_node_ids;

    ElementTopoHexa8 topology;
    double ex[num_elem_nodes], ey[num_elem_nodes], ez[num_elem_nodes];

    for(int ie=0; ie<num_elems-num_ghosted; ++ie){
        elem_node_ids = &element_node_conn[ie*num_elem_nodes];

        for(int j=0; j<num_elem_nodes; ++j){
            ex[j] = node_coordinates[elem_node_ids[j]*3];
            ey[j] = node_coordinates[elem_node_ids[j]*3+1];
            ez[j] = node_coordinates[elem_node_ids[j]*3+2];
        }
        double volume = topology.ComputeVolume(ex, ey, ez);
        cells[ie].GetVolume() = volume;
    }

}

void compute_cell_centroid(std::vector<Cell> & cells, std::vector<int> & element_node_conn, std::vector<double> & node_coordinates, int num_elems){
    const int num_elem_nodes = 8;
    int * elem_node_ids;
    double sumx, sumy, sumz;
    double centroid[3];

    for(int ie=0; ie<num_elems; ++ie){
        elem_node_ids = &element_node_conn[ie*num_elem_nodes];
        sumx = 0; sumy = 0; sumz = 0;

        for(int j=0; j<num_elem_nodes; ++j){
            sumx += node_coordinates[elem_node_ids[j]*3];
            sumy += node_coordinates[elem_node_ids[j]*3+1];
            sumz += node_coordinates[elem_node_ids[j]*3+2];
        }
        centroid[0] = sumx/num_elem_nodes;
        centroid[1] = sumy/num_elem_nodes;
        centroid[2] = sumz/num_elem_nodes;
        cells[ie].SetCoords(centroid);
    }
}

void delete_ghosted_faces(std::vector<Face> & all_faces, int ghosted_elem_index){
  std::vector<Face>::iterator iter;
  std::vector<Face> owned_faces;
  for (iter = all_faces.begin(); iter != all_faces.end();) {
    if (iter->GetElem2()==-1 && iter->GetElem1()>=ghosted_elem_index){
      //Add face to boundary faces
      ++iter;
    }
    else{
      owned_faces.push_back(*iter);
      ++iter;
    }
  }
  std::swap(all_faces, owned_faces);
}

void extract_BC_faces(std::vector<Face> & all_faces, std::vector<Face> & bc_faces){
  std::vector<Face>::iterator iter;
  std::vector<Face> internal_faces;
  for (iter = all_faces.begin(); iter != all_faces.end();) {
    if (iter->GetElem2()==-1){
      bc_faces.push_back(*iter);
      ++iter;
    }
    else{
      internal_faces.push_back(*iter);
      ++iter;
    }
  }
  std::swap(all_faces, internal_faces);
}

void organize_BC_faces(std::vector<Face> & all_faces, std::vector<Face> & bc_faces, std::set<int> & bc_nodes){
  std::vector<Face>::iterator iter;
  std::set<int>::iterator set_iter;
  const std::set<int>::iterator set_iter_end = bc_nodes.end();

  for (iter = all_faces.begin(); iter != all_faces.end();) {
    bool in_boundary = true;
    const std::vector<int> & face_nodes = iter->GetFaceNodes();
    std::vector<int>::const_iterator node_iter = face_nodes.begin();
    const std::vector<int>::const_iterator node_iter_end = face_nodes.end();
    for(; node_iter != node_iter_end; ++node_iter) {
      set_iter = bc_nodes.find(*node_iter);
      if(set_iter == set_iter_end){
        in_boundary=false;
        break;
      }
    }
    if (in_boundary){
      bc_faces.push_back(*iter);
      ++iter;
    }
    else
      ++iter;
  }
}
