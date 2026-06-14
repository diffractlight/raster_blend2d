cmake_minimum_required(VERSION 3.21)

project(raster_blend2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(BLEND2D_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/blend2d")
if(NOT EXISTS "${BLEND2D_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "Blend2D source is required at extern/blend2d. "
    "Add it with: git clone --depth 1 https://github.com/blend2d/blend2d extern/blend2d")
endif()

set(BLEND2D_STATIC TRUE CACHE BOOL "Build Blend2D as a static library" FORCE)
set(BLEND2D_NO_INSTALL TRUE CACHE BOOL "Do not install Blend2D artifacts into Python wheels" FORCE)
add_subdirectory("${BLEND2D_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/blend2d")

pybind11_add_module(_core MODULE src/raster_blend2d_native/bindings.cpp)
target_include_directories(_core PRIVATE "${BLEND2D_DIR}/blend2d")
target_link_libraries(_core PRIVATE blend2d::blend2d)

install(TARGETS _core LIBRARY DESTINATION raster_blend2d)
