################################################################################
# Project includes/libs
################################################################################

link_directories(${MPI_LIBRARY_DIRS})
include_directories(${MPI_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

FILE(GLOB HEADERS ${PROJECT_SOURCE_DIR}/include/*.h)
FILE(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.C)

add_library(miniaero ${SOURCES} ${HEADERS})

################################################################################
# Project linking
################################################################################

ADD_DEFINITIONS( -DATOMICS_FLUX)
#ADD_DEFINITIONS( -DCELL_FLUX)


IF( TPL_ENABLE_CUDA )
  ADD_DEFINITIONS( -DHAVE_CUDA )
  find_package(CUDA)
  cuda_add_executable(miniaero.exe ${SOURCES} ${HEADERS} test_cuda.cu)
ELSE()
add_executable(miniaero.exe ${SOURCES} ${HEADERS})
ENDIF()

target_link_libraries(miniaero.exe miniaero 
                      ${KOKKOS_LIBRARIES})

################################################################################
# Project install
################################################################################

install(PROGRAMS ${CMAKE_BINARY_DIR}/src/miniaero.exe DESTINATION bin)

#message("-- Installed executable name: ${SPARC_EXE}")
