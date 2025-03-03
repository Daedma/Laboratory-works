cmake_minimum_required(VERSION 3.14)
project(FFTExample CXX)

# Указываем стандарт C++ (например, C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Добавляем текущую директорию в путь для поиска заголовочных файлов
include_directories(${CMAKE_SOURCE_DIR})

# Находим FFTW
find_package(FFTW REQUIRED)
include_directories(${FFTW_INCLUDE_DIRS})

# Находим matplot++
# Предполагаем, что matplot++ находится в текущей директории
set(MATPLOTPP_DIR ${CMAKE_SOURCE_DIR}/matplotplusplus)
include_directories(${MATPLOTPP_DIR}/include)

# Если matplot++ требует дополнительные зависимости, добавьте их здесь
# Например, для OpenCV или других библиотек

# Добавляем исполняемый файл
add_executable(FFTExample main.cpp)

# Подключаем FFTW
target_link_libraries(FFTExample ${FFTW_LIBRARIES})

# Подключаем matplot++
target_link_libraries(FFTExample ${MATPLOTPP_DIR}/lib/libmatplot.a)

# Если matplot++ требует дополнительные библиотеки, добавьте их здесь
# Например:
# target_link_libraries(FFTExample opencv_core opencv_imgproc)