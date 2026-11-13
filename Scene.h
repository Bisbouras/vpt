#pragma once

#include "Common.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <stack>

using glm::vec3;
using glm::vec4;
using glm::mat4;
using glm::mat3;

enum Integrator { RAYTRACER, ANALYTICDIRECT, DIRECT, PATHTRACER };
enum ImportanceSampling { HEMISPHERE, COSINE, BRDF };
enum BrdfType { PHONG, GGX };
enum NEEMode { NEE_OFF, NEE_ON, NEE_MIS };

struct Ray {
    vec3 origin;
    vec3 direction;
    HD Ray(vec3 o, vec3 d) : origin(o), direction(glm::normalize(d)) {}
};

struct Material {
    vec3 diffuse = vec3(0);
    vec3 specular = vec3(0);
    vec3 emission = vec3(0);
    vec3 ambient = vec3(0.2f);
    float shininess = 0;
    float roughness = 1.0f;
    BrdfType brdf = PHONG;
};

struct Sphere {
    vec3 center;
    float radius;
    mat4 transform;
    mat4 invTransform;
    Material mat;
};

struct Triangle {
    vec3 v0, v1, v2;
    vec3 normal;
    Material mat;
};

struct DirLight {
    vec3 dir;
    vec3 col;
};

struct PointLight {
    vec3 pos;
    vec3 col;
};

struct QuadLight {
    vec3 a, ab, ac;   // corners: a, a+ab, a+ab+ac, a+ac
    vec3 intensity;   // radiance
};

// homogeneous global medium. sigma_t = sigma_a + sigma_s.
// g is the henyey-greenstein asymmetry parameter, -1 backward .. 0 isotropic .. 1 forward.
struct Medium {
    vec3 sigma_a = vec3(0);
    vec3 sigma_s = vec3(0);
    float g = 0.0f;
};

struct Scene {
    int w = 640, h = 480;
    int maxdepth = 5;
    std::string outfile = "raytrace.png";

    vec3 eye, center, up;
    float fovy = 90.0f;

    std::vector<Sphere> spheres;
    std::vector<Triangle> tris;

    std::vector<DirLight> dirLights;
    std::vector<PointLight> ptLights;

    vec3 atten = vec3(1, 0, 0); // const, linear, quad

    Integrator integrator = RAYTRACER;
    std::vector<QuadLight> quadLights;
    int lightsamples = 1;
    bool lightstratify = false;

    // path tracer
    int spp = 1;
    NEEMode nexteventestimation = NEE_OFF;
    bool russianroulette = false;
    ImportanceSampling importancesampling = HEMISPHERE;
    float gamma = 1.0f;
    unsigned seed = 0;   // rng - added to per-pixel seed as seed * w * h, 0 keeps the original images

    Medium globalMedium;
};

struct Hit {
    float t = -1;
    vec3 pos;
    vec3 norm;
    Material mat;
    bool found = false;
};