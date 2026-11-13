// GpuRender.h - host-facing entry points of the CUDA path tracer (no CUDA types, so main.cpp can include it)
#pragma once

#include "SceneView.h"
#include <string>

struct GpuTimes {
    double upload_ms = 0;     // scene arrays + zeroed sum buffer to the device
    double render_ms = 0;     // all sample launches, cuda events
    double download_ms = 0;   // sum buffer back to the host
};

// gpuInit - creates the cuda context (kept out of the timings), returns the device name
std::string gpuInit();

// renderGPU - path traces spp samples per pixel, one kernel launch per sample, and writes the
// per-pixel sums into sum (w * h, host memory), in the same order as the cpu (s = 0..spp-1)
void renderGPU(const SceneView& sc, int spp, vec3* sum, GpuTimes& t);
