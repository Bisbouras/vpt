// Kernels.cu - CUDA path tracing kernel, scene upload and launch loop
#include "GpuRender.h"
#include "CudaCheck.h"
#include "Raytracer.h"

#include <chrono>
#include <vector>

// renderKernel - one thread per pixel, adds sample s into its pixel's sum
__global__ void renderKernel(SceneView sc, int s, vec3* sum) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= sc.w || y >= sc.h) return;
    sum[y * sc.w + x] += renderPixel(sc, x, y, s);
}

namespace {

double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// upload - device copy of n elements (null when empty), remembered in owned for freeing
template <class T>
const T* upload(const T* src, int n, std::vector<void*>& owned) {
    if (n <= 0 || !src) return nullptr;
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, sizeof(T) * n));
    CUDA_CHECK(cudaMemcpy(d, src, sizeof(T) * n, cudaMemcpyHostToDevice));
    owned.push_back(d);
    return d;
}

}

std::string gpuInit() {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return std::string(prop.name) + " (sm_" + std::to_string(prop.major) + std::to_string(prop.minor) + ")";
}

void renderGPU(const SceneView& sc, int spp, vec3* sum, GpuTimes& t) {
    std::vector<void*> owned;
    size_t npix = (size_t)sc.w * sc.h;

    // upload - same view with device pointers. dir and point lights are whitted-only, left out
    auto t0 = std::chrono::steady_clock::now();
    SceneView dv = sc;
    dv.spheres = upload(sc.spheres, sc.numSpheres, owned);
    dv.tris = upload(sc.tris, sc.numTris, owned);
    dv.materials = upload(sc.materials, sc.numMaterials, owned);
    dv.nodes = upload(sc.nodes, sc.numNodes, owned);
    dv.quadLights = upload(sc.quadLights, sc.numQuadLights, owned);
    dv.dirLights = nullptr;
    dv.numDirLights = 0;
    dv.ptLights = nullptr;
    dv.numPtLights = 0;
    vec3* dsum = nullptr;
    CUDA_CHECK(cudaMalloc(&dsum, sizeof(vec3) * npix));
    CUDA_CHECK(cudaMemset(dsum, 0, sizeof(vec3) * npix));
    owned.push_back(dsum);
    CUDA_CHECK(cudaDeviceSynchronize());
    t.upload_ms = msSince(t0);

    // render - one launch per sample, 16x16 blocks
    dim3 block(16, 16);
    dim3 grid((sc.w + block.x - 1) / block.x, (sc.h + block.y - 1) / block.y);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int s = 0; s < spp; s++) {
        renderKernel<<<grid, block>>>(dv, s, dsum);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    t.render_ms = ms;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // download
    t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(sum, dsum, sizeof(vec3) * npix, cudaMemcpyDeviceToHost));
    t.download_ms = msSince(t0);

    for (void* p : owned)
        CUDA_CHECK(cudaFree(p));
}
