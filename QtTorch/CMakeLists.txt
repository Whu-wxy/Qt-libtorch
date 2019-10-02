cmake_minimum_required(VERSION 2.8)

project(QtTorch)

#如果要生成debug版本
SET(CMAKE_BUILD_TYPE "Debug")
#如果要生成release版本
#SET(CMAKE_BUILD_TYPE "Release")
#添加的支持debug和release支持
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(Qt5Widgets_DIR /opt/Qt5.13.0/5.13.0/gcc_64/lib/cmake/Qt5Widgets)
set(Qt5Gui_DIR /opt/Qt5.13.0/5.13.0/gcc_64/lib/cmake/Qt5Gui)
set(Qt5Core_DIR /opt/Qt5.13.0/5.13.0/gcc_64/lib/cmake/Qt5Core)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Core REQUIRED)
qt5_wrap_cpp( MOC torchtestdlg.h)

find_package(Torch REQUIRED)
include_directories(  ${TORCH_INCLUDE_DIRS} )

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(QtTorch main.cpp torchtestdlg.cpp torchtestdlg.h)
target_link_libraries(QtTorch Qt5::Widgets)

target_link_libraries(QtTorch ${OpenCV_LIBS} ${TORCH_LIBRARIES}) #add_executable then target_link_libraries

