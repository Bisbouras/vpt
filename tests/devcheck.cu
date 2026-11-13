// devcheck.cu - compiles the shared path tracer headers for the gpu (run by scripts/devcheck.py, not built by cmake)
#include "Raytracer.h"

// devcheck - one call into pathTrace pulls in every function the path tracer reaches
__global__ void devcheck(const Scene* sc, const BVH* bvh, std::mt19937* rng, vec3* out) {
    Ray r = makeRay(0.5f, 0.5f, *sc);
    out[0] = pathTrace(r, *bvh, *sc, *rng);
}
