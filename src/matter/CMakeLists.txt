cmake_minimum_required(VERSION 3.18)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
project(IMSRG-neutrino CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -g")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address")


if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
  include_directories("/opt/homebrew/opt/libomp/include")
endif()
include_directories(src)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GSL REQUIRED)
find_package(OpenMP REQUIRED)

set(MAIN_SOURCES
  src/main.cc
  src/Profiler.cc
  src/BasicFunctions.cc
  src/NdSpline.cc
  src/NNPartialWave.cc
  src/NNPWtoGrid.cc
  src/TwoBodyChannel.cc
  src/TwoBodySpace.cc
  src/ModelSpace.cc
  src/OneBodyOperator.cc
  src/TwoBodyOperator.cc
  src/Operator.cc
  src/ChEFT.cc
  src/MBPT.cc
  src/Commutator.cc
  src/IMSRG.cc
)

add_executable(imsrg_matter.exe ${MAIN_SOURCES})
target_link_libraries(
  imsrg_matter.exe
  Eigen3::Eigen
  GSL::gsl
  OpenMP::OpenMP_CXX
)
