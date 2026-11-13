// Common.h - definitions shared by cpu and gpu code
#pragma once

// hd - host + device under nvcc, nothing for the cpu compiler
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
