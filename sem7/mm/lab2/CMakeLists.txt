cmake_minimum_required(VERSION 3.26.4)

project(queueing)

find_package(SDL2 CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(imgui CONFIG REQUIRED)

add_executable(queueing main.cpp View.cpp QueueingModel.cpp)

target_link_libraries(queueing
        PRIVATE
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
    )

target_link_libraries(queueing PRIVATE Vulkan::Vulkan)

target_link_libraries(queueing PRIVATE imgui::imgui)