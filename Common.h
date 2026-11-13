// Common.h - definitions shared by cpu and gpu code
#pragma once

// hd - host + device under nvcc, nothing for the cpu compiler
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// maxf - same as std::max, b only when a < b (so a nan in a is kept, unlike fmaxf)
HD inline float maxf(float a, float b) { return a < b ? b : a; }
