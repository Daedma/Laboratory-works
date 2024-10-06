cmake_minimum_required(VERSION 3.26.4)

project(planets)

find_package(SDL2 CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS qvm)

add_executable(planets main.cpp View.cpp)

# target_include_directories(planets PUBLIC "${IMGUI_DIR}" "${IMGUI_DIR}/backends")

target_link_libraries(planets
        PRIVATE
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
    )

target_link_libraries(planets PRIVATE Vulkan::Vulkan)

target_link_libraries(planets PRIVATE imgui::imgui)

target_link_libraries(planets PRIVATE Boost::qvm)