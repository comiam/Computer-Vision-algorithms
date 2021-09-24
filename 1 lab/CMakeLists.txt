cmake_minimum_required(VERSION 3.16)
project(1_lab)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5Charts REQUIRED)

add_executable(1_lab main.cpp MainWindow.cpp ImageViewer.cpp HsvSliders.cpp HistogramWindow.cpp HistogramWindow.cpp)

target_link_libraries(1_lab Qt5::Core)
target_link_libraries(1_lab Qt5::Widgets)
target_link_libraries(1_lab Qt5::Charts)