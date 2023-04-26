cmake_minimum_required(VERSION 3.21.1)

project(lab6)

set(CMAKE_CXX_STANDARD 20)

add_executable(server Server.cpp ServerApplication.cpp)
target_include_directories(server PUBLIC "C:\\Libs\\SFML-2.5.1\\include")
target_link_directories(server PUBLIC "C:\\Libs\\SFML-2.5.1\\sfml-builds\\msvc\\lib\\Debug"
									  "C:\\Libs\\SFML-2.5.1\\extlibs\\libs-msvc-universal\\x64")
target_link_libraries(server 
				"ws2_32" 
				"sfml-system-d" 
				"sfml-network-d")

add_executable(client Client.cpp ClientApplication.cpp)
target_include_directories(client PUBLIC "C:\\Libs\\SFML-2.5.1\\include")
target_link_directories(client PUBLIC "C:\\Libs\\SFML-2.5.1\\sfml-builds\\msvc\\lib\\Debug"
									  "C:\\Libs\\SFML-2.5.1\\extlibs\\libs-msvc-universal\\x64")
target_link_libraries(client 
				"ws2_32" 
				"sfml-system-d" 
				"sfml-network-d")