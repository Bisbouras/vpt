// devcheck.cu - compiles the shared path tracer headers for the gpu (run by scripts/devcheck.py, not built by cmake)
#include "Raytracer.h"

// devcheck - one call into renderPixel pulls in every function the path tracer reaches
__global__ void devcheck(SceneView sc, vec3* out) {
    out[0] += renderPixel(sc, 0, 0, 1);
}
