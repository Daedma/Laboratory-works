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
	ellipse.cpp)

add_executable(intersection intersection.cpp ${CLASSES_SRC})

target_link_libraries(intersection PUBLIC board)

target_include_directories(intersection PRIVATE ${SOURCE_DIR} libboard/include)


add_executable(reflection reflection.cpp ${CLASSES_SRC})

target_link_libraries(reflection PUBLIC board)

target_include_directories(reflection PRIVATE ${SOURCE_DIR} libboard/include)


add_executable(refraction refraction.cpp ${CLASSES_SRC})

target_link_libraries(refraction PUBLIC board)

target_include_directories(refraction PRIVATE ${SOURCE_DIR} libboard/include)


add_executable(demonstration demonstration.cpp ${CLASSES_SRC})

target_link_libraries(demonstration PUBLIC board)

target_include_directories(demonstration PRIVATE  ${SOURCE_DIR} libboard/include)