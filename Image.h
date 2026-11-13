// Image.h - output pass and float image output (pfm)
#pragma once

#include "Scene.h"
#include <cmath>
#include <fstream>
#include <string>

// outputPass - per-pixel sums of n samples -> linear average (for the pfm) and 8-bit rgb.
// 8-bit is clamp to [0,1], gamma when it isn't 1, then truncate (not round), as the renderer always did
inline void outputPass(const vec3* sum, int n, int w, int h, float gamma, vec3* avg, unsigned char* pixels) {
    for (int p = 0; p < w * h; p++) {
        vec3 col = sum[p] / (float)n;
        avg[p] = col;
        col = glm::clamp(col, 0.0f, 1.0f);
        if (gamma != 1.0f) {
            float inv = 1.0f / gamma;
            col.r = powf(col.r, inv);
            col.g = powf(col.g, inv);
            col.b = powf(col.b, inv);
        }
        pixels[p * 3] = (unsigned char)(col.r * 255);
        pixels[p * 3 + 1] = (unsigned char)(col.g * 255);
        pixels[p * 3 + 2] = (unsigned char)(col.b * 255);
    }
}

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
