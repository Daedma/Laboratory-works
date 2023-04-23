cmake_minimum_required(VERSION 3.21.1)

project(lab5)

set(CMAKE_CXX_STANDARD 17)

add_executable(server server.cpp)
target_link_libraries(server ws2_32)

add_executable(client client.cpp)
target_link_libraries(client ws2_32)