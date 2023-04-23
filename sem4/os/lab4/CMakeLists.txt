cmake_minimum_required(VERSION 3.21.1)

project(lab4)

set(CMAKE_CXX_STANDARD 17)

add_executable(server server.cpp)

add_executable(client client.cpp)