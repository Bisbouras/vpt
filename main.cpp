#include <iostream>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Scene.h"
#include "Parser.h"
#include "Raytracer.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: raytracer <scenefile>" << std::endl;
        return 1;
    }

    std::cout << "loading " << argv[1] << std::endl;
    Scene scene = parseScene(argv[1]);

    std::cout << "integrator=" << scene.integrator
          << " quads=" << scene.quadLights.size()
          << " samples=" << scene.lightsamples
          << " stratify=" << scene.lightstratify
          << " spp=" << scene.spp
          << " nee=" << scene.nexteventestimation
          << " rr=" << scene.russianroulette
          << " maxdepth=" << scene.maxdepth << std::endl;

    std::cout << scene.w << "x" << scene.h
              << " | " << scene.spheres.size() << " spheres"
              << " | " << scene.tris.size() << " tris" << std::endl;

    std::vector<unsigned char> pixels(scene.w * scene.h * 3, 0);
    render(scene, pixels.data());

    stbi_write_png(scene.outfile.c_str(), scene.w, scene.h, 3, pixels.data(), scene.w * 3);
    std::cout << "wrote " << scene.outfile << std::endl;

    return 0;
}