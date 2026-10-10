cmake_minimum_required(VERSION 3.15)
project(Raytracer)

set(CMAKE_CXX_STANDARD 17)

# Find GLM (header-only)
find_package(glm CONFIG QUIET)
if(NOT glm_FOUND)
    # Fallback: assume glm is in include/ directory
    include_directories(${CMAKE_SOURCE_DIR}/include)
endif()

# Include local headers (stb_image_write.h goes here too)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

add_executable(raytracer main.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(raytracer PUBLIC OpenMP::OpenMP_CXX)

# If GLM was found via package manager
if(glm_FOUND)
    target_link_libraries(raytracer glm::glm)
endif()
