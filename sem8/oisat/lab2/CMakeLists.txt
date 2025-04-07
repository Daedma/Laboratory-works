cmake_minimum_required(VERSION 3.26)
project(LR1 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS OFF)

include_directories(${CMAKE_SOURCE_DIR})

set(CMAKE_DISABLE_FIND_PACKAGE_ImageMagick)

add_subdirectory(libboard)

set(LIBBOARD_ROOT ${CMAKE_SOURCE_DIR}/libboards)

add_subdirectory(glm)

set(CLASSES_SRC 
	plane.cpp
	shape.cpp
	sphere.cpp
	ellipse.cpp
	biconvex_lens.cpp)

add_executable(focus focus.cpp ${CLASSES_SRC})

target_link_libraries(focus PUBLIC board)

target_include_directories(focus PRIVATE ${SOURCE_DIR} libboard/include)


add_executable(lens lens.cpp ${CLASSES_SRC})

target_link_libraries(lens PUBLIC board)

target_include_directories(lens PRIVATE ${SOURCE_DIR} libboard/include)