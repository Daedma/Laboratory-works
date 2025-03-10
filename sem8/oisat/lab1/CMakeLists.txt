cmake_minimum_required(VERSION 3.26)
project(LR3 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS OFF)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(fftw)

set(FFTW_ROOT ${CMAKE_SOURCE_DIR}/fftw)

add_subdirectory(matplotplusplus)

add_executable(dark_field dark_field.cpp LightField.cpp)

target_link_libraries(dark_field PUBLIC fftw3 matplot)

target_include_directories(dark_field PRIVATE fftw/api ${SOURCE_DIR})

add_executable(zernick zernick.cpp LightField.cpp)

target_link_libraries(zernick PUBLIC fftw3 matplot)

target_include_directories(zernick PRIVATE fftw/api ${SOURCE_DIR})