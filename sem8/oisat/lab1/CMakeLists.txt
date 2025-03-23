cmake_minimum_required(VERSION 3.26)
project(LR1 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS OFF)

include_directories(${CMAKE_SOURCE_DIR})

set(CMAKE_DISABLE_FIND_PACKAGE_ImageMagick)

add_subdirectory(libboard)

add_subdirectory(glm)

add_executable(intersection intersection.cpp)

target_include_directories(intersection PRIVATE ${SOURCE_DIR})


add_executable(reflection reflection.cpp)

target_include_directories(reflection PRIVATE ${SOURCE_DIR})


add_executable(refraction refraction.cpp)

target_include_directories(refraction PRIVATE ${SOURCE_DIR})


add_executable(demonstration demonstration.cpp)

target_link_libraries(demonstration PUBLIC board)

target_include_directories(demonstration PRIVATE  ${SOURCE_DIR})