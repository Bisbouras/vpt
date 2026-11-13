#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
#include <omp.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Scene.h"
#include "Parser.h"
#include "Raytracer.h"
#include "Image.h"

static const char* USAGE =
    "usage: raytracer <scenefile> [--spp N] [--out file.png] [--seed N] [--quiet]";

static double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// cli - whole-string integer parse, false on junk or out of range
static bool parseInt(const char* s, long lo, long hi, long& out) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < lo || v > hi) return false;
    out = v;
    return true;
}

int main(int argc, char* argv[]) {
    std::string sceneFile, outFile;
    long sppOverride = 0, seed = 0;
    bool quiet = false;

    for (int a = 1; a < argc; a++) {
        std::string arg = argv[a];
        bool hasVal = a + 1 < argc;
        if (arg == "--spp" && hasVal) {
            if (!parseInt(argv[++a], 1, 1L << 30, sppOverride)) {
                std::cerr << "bad --spp value " << argv[a] << std::endl;
                return 1;
            }
        }
        else if (arg == "--seed" && hasVal) {
            if (!parseInt(argv[++a], 0, 1L << 20, seed)) {
                std::cerr << "bad --seed value " << argv[a] << std::endl;
                return 1;
            }
        }
        else if (arg == "--out" && hasVal) {
            outFile = argv[++a];
        }
        else if (arg == "--quiet") {
            quiet = true;
        }
        else if (arg.rfind("--", 0) == 0 || !sceneFile.empty()) {
            std::cerr << "unexpected argument " << arg << "\n" << USAGE << std::endl;
            return 1;
        }
        else {
            sceneFile = arg;
        }
    }
    if (sceneFile.empty()) {
        std::cerr << USAGE << std::endl;
        return 1;
    }

    std::cout << "loading " << sceneFile << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    Scene scene = parseScene(sceneFile);
    double parseMs = msSince(t0);

    if (sppOverride > 0) scene.spp = (int)sppOverride;
    if (!outFile.empty()) scene.outfile = outFile;
    scene.seed = (unsigned)seed;

    std::cout << "integrator=" << scene.integrator
          << " quads=" << scene.quadLights.size()
          << " samples=" << scene.lightsamples
          << " stratify=" << scene.lightstratify
          << " spp=" << scene.spp
          << " nee=" << scene.nexteventestimation
          << " rr=" << scene.russianroulette
          << " maxdepth=" << scene.maxdepth
          << " seed=" << scene.seed << std::endl;

    std::cout << scene.w << "x" << scene.h
              << " | " << scene.spheres.size() << " spheres"
              << " | " << scene.tris.size() << " tris"
              << " | " << scene.materials.size() << " materials" << std::endl;

    t0 = std::chrono::steady_clock::now();
    BVH bvh;
    bvh.build(scene);
    bvh.reorder(scene);
    double bvhMs = msSince(t0);
    std::cout << "BVH: " << bvh.nodes.size() << " nodes, "
              << bvh.prims.size() << " prims" << std::endl;

    std::vector<unsigned char> pixels(scene.w * scene.h * 3, 0);
    std::vector<vec3> accum(scene.w * scene.h, vec3(0));
    t0 = std::chrono::steady_clock::now();
    render(scene, bvh, pixels.data(), accum.data(), quiet);
    double renderMs = msSince(t0);

    // timing - one line for scripts/bench.py
    double msamples = (double)scene.w * scene.h * scene.spp / (renderMs / 1000.0) / 1e6;
    std::cout << std::fixed << std::setprecision(1)
              << "time parse_ms=" << parseMs
              << " bvh_ms=" << bvhMs
              << " render_ms=" << renderMs
              << " spp=" << scene.spp
              << " threads=" << omp_get_max_threads()
              << std::setprecision(3) << " msamples_s=" << msamples << std::endl;

    if (!stbi_write_png(scene.outfile.c_str(), scene.w, scene.h, 3, pixels.data(), scene.w * 3)) {
        std::cerr << "cant write " << scene.outfile << std::endl;
        return 1;
    }
    std::string pfm = pfmPath(scene.outfile);
    if (!writePFM(pfm, scene.w, scene.h, accum.data())) {
        std::cerr << "cant write " << pfm << std::endl;
        return 1;
    }
    std::cout << "wrote " << scene.outfile << " and " << pfm << std::endl;

    return 0;
}
