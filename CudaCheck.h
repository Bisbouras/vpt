// CudaCheck.h - CUDA_CHECK macro: on any CUDA runtime error, report call, file and line, then exit
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "cuda error %s (%s)\n  in %s\n  at %s:%d\n", cudaGetErrorName(err_), \
                cudaGetErrorString(err_), #call, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)
