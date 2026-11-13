cmake_minimum_required(VERSION 3.18)
project(Raytracer LANGUAGES CXX)

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

# cuda backend - optional, adds --gpu. without a cuda compiler (or with -DVPT_CUDA=OFF) the build is cpu only
option(VPT_CUDA "Build the CUDA path tracer (--gpu) when a CUDA compiler is found" ON)
if(VPT_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        target_sources(raytracer PRIVATE Kernels.cu)
        # rtx 3070 ti - ampere, sm_86
        set_target_properties(raytracer PROPERTIES CUDA_ARCHITECTURES 86)
        # glm calls constexpr host functions (std::multiplies) from device code
        target_compile_options(raytracer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        target_compile_definitions(raytracer PRIVATE VPT_HAS_CUDA)
        message(STATUS "CUDA backend on (${CMAKE_CUDA_COMPILER})")
    else()
        message(STATUS "no CUDA compiler found, building the CPU renderer only")
    endif()
endif()
