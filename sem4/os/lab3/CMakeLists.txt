cmake_minimum_required(VERSION 3.16.3)

project(Lab3 VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(main main.cpp)

add_executable(server ServerApplication.cpp)

add_executable(client ClientApplication.cpp)