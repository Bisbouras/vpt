// SceneView.h - plain view of the scene and bvh for rendering: raw pointers, counts and settings, no stl
#pragma once

#include "Scene.h"
#include "BVH.h"

// sceneview - field names match Scene so shading code reads the same, arrays become pointer + count.
// points into the Scene and BVH vectors, which must outlive it
struct SceneView {
    // camera
    int w, h;
    vec3 eye, center, up;
    float fovy;

    // geometry and materials
    const Sphere* spheres;     int numSpheres;
    const Triangle* tris;      int numTris;
    const Material* materials; int numMaterials;
    const BVHNode* nodes;      int numNodes;
    const Prim* prims;         int numPrims;

    // lights - dir and point lights are only read by the cpu-only whitted integrator
    const QuadLight* quadLights;  int numQuadLights;
    const DirLight* dirLights;    int numDirLights;
    const PointLight* ptLights;   int numPtLights;
    vec3 atten;

    // integrator settings
    int maxdepth;
    int lightsamples;
    bool lightstratify;
    NEEMode nexteventestimation;
    bool russianroulette;
    ImportanceSampling importancesampling;
    Medium globalMedium;
};

inline SceneView makeView(const Scene& sc, const BVH& bvh) {
    SceneView v;
    v.w = sc.w;
    v.h = sc.h;
    v.eye = sc.eye;
    v.center = sc.center;
    v.up = sc.up;
    v.fovy = sc.fovy;

    v.spheres = sc.spheres.data();       v.numSpheres = (int)sc.spheres.size();
    v.tris = sc.tris.data();             v.numTris = (int)sc.tris.size();
    v.materials = sc.materials.data();   v.numMaterials = (int)sc.materials.size();
    v.nodes = bvh.nodes.data();          v.numNodes = (int)bvh.nodes.size();
    v.prims = bvh.prims.data();          v.numPrims = (int)bvh.prims.size();

    v.quadLights = sc.quadLights.data(); v.numQuadLights = (int)sc.quadLights.size();
    v.dirLights = sc.dirLights.data();   v.numDirLights = (int)sc.dirLights.size();
    v.ptLights = sc.ptLights.data();     v.numPtLights = (int)sc.ptLights.size();
    v.atten = sc.atten;

    v.maxdepth = sc.maxdepth;
    v.lightsamples = sc.lightsamples;
    v.lightstratify = sc.lightstratify;
    v.nexteventestimation = sc.nexteventestimation;
    v.russianroulette = sc.russianroulette;
    v.importancesampling = sc.importancesampling;
    v.globalMedium = sc.globalMedium;
    return v;
}
