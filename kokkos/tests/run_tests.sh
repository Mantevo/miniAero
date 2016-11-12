#!/bin/bash

green='\e[0;32m'
red='\e[0;31m'
NC='\e[0m'

EXE=`echo "$(cd "$(dirname "$1")"; pwd)/$(basename "$1")"`
DO_MPI=`echo $2 | grep MPI | wc -l`

cd 3D_Sod_Serial
./3d_sod_serial_test.sh ${EXE}
if [ "$?" -ne 0 ];
then
  echo -e "${red}3D_Sod_Serial Test Failed${NC}"
else
  echo -e "${green} 3D_Sod_Serial Test Passed${NC}"
fi 
cd ..

if [ "${DO_MPI}" -eq 1 ];
then
cd 3D_Sod_Parallel
./3d_sod_parallel_test.sh ${EXE}
if [ "$?" -ne 0 ];
then
  echo -e "${red}3D_Sod_Parallel Test Failed${NC}"
else
  echo -e "${green} 3D_Sod_Parallel Test Passed${NC}"
fi
cd ..
fi

cd Ramp_Serial 
./ramp_serial_test.sh ${EXE}
if [ "$?" -ne 0 ];
then
  echo -e "${red}Ramp_Serial Test Failed${NC}"
else
  echo -e "${green} Ramp_Serial Test Passed${NC}"
fi
cd ..

if [ "${DO_MPI}" -eq 1 ];
then
cd Ramp_Parallel
./ramp_parallel_test.sh ${EXE}
if [ "$?" -ne 0 ];
then
  echo -e "${red}Ramp_Parallel Test Failed${NC}"
else
  echo -e "${green} Ramp_Parallel Test Passed${NC}"
fi
cd ..
fi

cd FlatPlate_Serial 
./flatplate_serial_test.sh ${EXE}
if [ "$?" -ne 0 ];
then
  echo -e "${red}FlatPlate_Serial Test Failed${NC}"
else
  echo -e "${green} FlatPlate_Serial Test Passed${NC}"
fi
cd ..

if [ "${DO_MPI}" -eq 1 ];
then
cd FlatPlate_Parallel
./flatplate_parallel_test.sh ${EXE}
if [ "$?" -ne 0 ];
then
  echo -e "${red}FlatPlate_Parallel Test Failed${NC}"
else
  echo -e "${green} FlatPlate_Parallel Test Passed${NC}"
fi
cd ..
fi
