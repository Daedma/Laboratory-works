cmake_minimum_required(VERSION 3.26)
project(LR3 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -fopenmp=libomp")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp=libomp")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	include_directories(${OpenMP_INCLUDE_DIRS})
endif()

option(BUILD_SHARED_LIBS OFF)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(fftw)

set(FFTW_ROOT ${CMAKE_SOURCE_DIR}/fftw)

add_subdirectory(matplotplusplus)

add_executable(dark_field dark_field.cpp LightField.cpp)

target_link_libraries(dark_field PUBLIC fftw3 fftw3_omp matplot OpenMP::OpenMP_CXX)

target_include_directories(dark_field PRIVATE fftw/api ${SOURCE_DIR})


add_executable(zernick zernick.cpp LightField.cpp)

target_link_libraries(zernick PUBLIC fftw3 fftw3_omp matplot OpenMP::OpenMP_CXX)

target_include_directories(zernick PRIVATE fftw/api ${SOURCE_DIR})


add_executable(derivative derivative.cpp LightField.cpp)

target_link_libraries(derivative PUBLIC fftw3 fftw3_omp matplot OpenMP::OpenMP_CXX)

target_include_directories(derivative PRIVATE fftw/api ${SOURCE_DIR})