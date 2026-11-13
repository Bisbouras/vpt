// Image.h - float image output (pfm)
#pragma once

#include "Scene.h"
#include <fstream>
#include <string>

// pfm - "PF" colour, negative scale = little-endian, rows stored bottom to top
inline bool writePFM(const std::string& path, int w, int h, const vec3* data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "PF\n" << w << " " << h << "\n-1.0\n";
    for (int j = h - 1; j >= 0; j--) {
        for (int i = 0; i < w; i++) {
            const vec3& c = data[j * w + i];
            float px[3] = { c.r, c.g, c.b };
            f.write(reinterpret_cast<const char*>(px), sizeof(px));
        }
    }
    return (bool)f;
}

// pfm path next to the png - swaps the extension, or appends .pfm
inline std::string pfmPath(const std::string& png) {
    size_t slash = png.find_last_of("/\\");
    size_t dot = png.rfind('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
        return png + ".pfm";
    return png.substr(0, dot) + ".pfm";
}
