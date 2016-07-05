project(kokkos_miniaero)

cmake_minimum_required(VERSION 2.6)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/kokkos/core/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/kokkos/core/src)

SET(HEADERS "")
SET(SOURCES "")

FILE(GLOB HEADERS kokkos/core/src/*.hpp)

SET(KOKKOS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/kokkos/core/src)
SET(KOKKOS_LIBRARIES)

###########################
# kokkos/core/src/Host/*
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/kokkos/core/src/impl)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/kokkos/core/src/impl)

SET(HEADERS "")
SET(SOURCES "")

FILE(GLOB HEADERS_THREADS kokkos/core/src/Threads/*.hpp)
FILE(GLOB SOURCES_THREADS kokkos/core/src/Threads/*.cpp)
FILE(GLOB HEADERS kokkos/core/src/impl/*.hpp)

#-----------------------------------------------------------------------------
# Sources always included

LIST(APPEND HEADERS_PRIVATE ${HEADERS_THREADS} )
LIST(APPEND SOURCES         ${SOURCES_THREADS} )
LIST(APPEND SOURCES kokkos/core/src/impl/Kokkos_HostSpace.cpp)

#-----------------------------------------------------------------------------
# Hardware locality detection functionality depends upon TPL availability
IF( TPL_ENABLE_HWLOC )
    #
    #  http://www.open-mpi.org/projects/hwloc/
    #  Version 1.3 or greater
    #
    #
#  LINK_DIRECTORIES(${HWLOC_LIB_DIRS})
  INCLUDE_DIRECTORIES(${HWLOC_INCLUDE_DIRS})
#ELSE()
#    LIST(APPEND SOURCES kokkos/core/src/Host/Kokkos_hwloc_unavailable.cpp)
ENDIF()
#-----------------------------------------------------------------------------
# Thread spawning functionality depend upon TPL availability
#IF( TPL_ENABLE_Pthread )
#    LIST(APPEND SOURCES kokkos/core/src/Kokkos_Threads.cpp)
#ELSEIF( TPL_ENABLE_Winthread )
#    LIST(APPEND SOURCES kokkos/core/src/Host/Kokkos_Host_winthread.cpp)
#ELSE()
#    LIST(APPEND SOURCES kokkos/core/src/Host/Kokkos_Host_nothread.cpp)
#ENDIF()

#-----------------------------------------------------------------------------
ADD_LIBRARY(
  kokkos_devicehost
  ${HEADERS}
  ${SOURCES}
  )
LIST(APPEND KOKKOS_LIBRARIES kokkos_devicehost)
 
###########################


###########################
# kokkos/core/src/impl/*
SET(HEADERS "")
SET(SOURCES "")

FILE(GLOB HEADERS kokkos/core/src/impl/*.hpp)
FILE(GLOB SOURCES kokkos/core/src/impl/*.cpp)

ADD_LIBRARY(
    kokkoscore_impl
    ${HEADERS}
    ${SOURCES}
    )
LIST(APPEND KOKKOS_LIBRARIES kokkoscore_impl)
###########################

IF(TPL_ENABLE_CUDA)
    find_package(CUDA)
    ###########################
    # kokkos/core/src/Cuda/*

    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/kokkos/core/src/Cuda/)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/kokkos/core/src/Cuda/)

    SET(HEADERS "")
    SET(SOURCES "")
    SET(CUDA_SOURCES "")

    FILE(GLOB HEADERS kokkos/core/src/Cuda/*.hpp)
    FILE(GLOB CUDA_SOURCES kokkos/core/src/Cuda/*.cu)

    set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

    CUDA_COMPILE(CUDA_OBJECTS ${CUDA_SOURCES})

    SET(SOURCES ${SOURCES} ${CUDA_OBJECTS})

    CUDA_ADD_LIBRARY(
        kokkoscore_devicecuda
        ${HEADERS}
        ${SOURCES}
    )
    LIST(APPEND KOKKOS_LIBRARIES kokkoscore_devicecuda)
    ###########################
ENDIF()
SET(HEADERS "")
SET(SOURCES "")

add_subdirectory(src)

