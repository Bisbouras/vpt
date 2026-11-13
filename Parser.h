#pragma once

#include "Scene.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stack>

// materials - index of m in sc.materials, appended if new (searched newest first)
inline int materialId(Scene& sc, const Material& m) {
    for (int i = (int)sc.materials.size() - 1; i >= 0; i--)
        if (sc.materials[i] == m) return i;
    sc.materials.push_back(m);
    return (int)sc.materials.size() - 1;
}

inline Scene parseScene(const std::string& filename) {
    Scene scene;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "cant open " << filename << std::endl;
        exit(1);
    }

    std::stack<mat4> matStack;
    mat4 curTransform = mat4(1.0f);
    Material curMat;

    std::vector<vec3> verts;
    std::vector<vec3> vnorms;
    std::vector<vec3> vnormdirs;

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd.empty()) continue;

        if (cmd == "size") {
            ss >> scene.w >> scene.h;
        }
        else if (cmd == "maxdepth") {
            ss >> scene.maxdepth;
        }
        else if (cmd == "output") {
            ss >> scene.outfile;
        }
        else if (cmd == "camera") {
            ss >> scene.eye.x >> scene.eye.y >> scene.eye.z;
            ss >> scene.center.x >> scene.center.y >> scene.center.z;
            ss >> scene.up.x >> scene.up.y >> scene.up.z;
            ss >> scene.fovy;
        }
        else if (cmd == "sphere") {
            Sphere s;
            ss >> s.center.x >> s.center.y >> s.center.z >> s.radius;
            s.transform = curTransform;
            s.invTransform = glm::inverse(curTransform);
            s.matId = materialId(scene, curMat);
            scene.spheres.push_back(s);
        }
        else if (cmd == "maxverts") {
            int n; ss >> n;
            verts.reserve(n);
        }
        else if (cmd == "maxvertnorms") {
            int n; ss >> n; // dont really need this
        }
        else if (cmd == "vertex") {
            vec3 v;
            ss >> v.x >> v.y >> v.z;
            verts.push_back(v);
        }
        else if (cmd == "vertexnormal") {
            vec3 v, n;
            ss >> v.x >> v.y >> v.z >> n.x >> n.y >> n.z;
            vnorms.push_back(v);
            vnormdirs.push_back(n);
        }
        else if (cmd == "tri") {
            int a, b, c;
            ss >> a >> b >> c;
            Triangle t;
            t.v0 = vec3(curTransform * vec4(verts[a], 1.0f));
            t.v1 = vec3(curTransform * vec4(verts[b], 1.0f));
            t.v2 = vec3(curTransform * vec4(verts[c], 1.0f));
            t.normal = glm::normalize(glm::cross(t.v1 - t.v0, t.v2 - t.v0));
            t.matId = materialId(scene, curMat);
            scene.tris.push_back(t);
        }
        else if (cmd == "trinormal") {
            int a, b, c;
            ss >> a >> b >> c;
            // just do flat shading for now
            Triangle t;
            t.v0 = vec3(curTransform * vec4(vnorms[a], 1.0f));
            t.v1 = vec3(curTransform * vec4(vnorms[b], 1.0f));
            t.v2 = vec3(curTransform * vec4(vnorms[c], 1.0f));
            t.normal = glm::normalize(glm::cross(t.v1 - t.v0, t.v2 - t.v0));
            t.matId = materialId(scene, curMat);
            scene.tris.push_back(t);
        }
        else if (cmd == "translate") {
            vec3 t;
            ss >> t.x >> t.y >> t.z;
            curTransform = curTransform * glm::translate(mat4(1.0f), t);
        }
        else if (cmd == "rotate") {
            float x, y, z, ang;
            ss >> x >> y >> z >> ang;
            curTransform = curTransform * glm::rotate(mat4(1.0f), glm::radians(ang), vec3(x, y, z));
        }
        else if (cmd == "scale") {
            vec3 s;
            ss >> s.x >> s.y >> s.z;
            curTransform = curTransform * glm::scale(mat4(1.0f), s);
        }
        else if (cmd == "pushTransform") {
            matStack.push(curTransform);
        }
        else if (cmd == "popTransform") {
            if (!matStack.empty()) {
                curTransform = matStack.top();
                matStack.pop();
            }
        }
        else if (cmd == "directional") {
            DirLight dl;
            ss >> dl.dir.x >> dl.dir.y >> dl.dir.z;
            ss >> dl.col.x >> dl.col.y >> dl.col.z;
            dl.dir = glm::normalize(dl.dir);
            scene.dirLights.push_back(dl);
        }
        else if (cmd == "point") {
            PointLight pl;
            ss >> pl.pos.x >> pl.pos.y >> pl.pos.z;
            ss >> pl.col.x >> pl.col.y >> pl.col.z;
            scene.ptLights.push_back(pl);
        }
        else if (cmd == "attenuation") {
            ss >> scene.atten.x >> scene.atten.y >> scene.atten.z;
        }
        else if (cmd == "ambient") {
            ss >> curMat.ambient.x >> curMat.ambient.y >> curMat.ambient.z;
        }
        else if (cmd == "diffuse") {
            ss >> curMat.diffuse.x >> curMat.diffuse.y >> curMat.diffuse.z;
        }
        else if (cmd == "specular") {
            ss >> curMat.specular.x >> curMat.specular.y >> curMat.specular.z;
        }
        else if (cmd == "shininess") {
            ss >> curMat.shininess;
        }
        else if (cmd == "emission") {
            ss >> curMat.emission.x >> curMat.emission.y >> curMat.emission.z;
        }
        else if (cmd == "integrator") {
            std::string name;
            ss >> name;
            if (name == "analyticdirect") scene.integrator = ANALYTICDIRECT;
            else if (name == "direct") scene.integrator = DIRECT;
            else if (name == "pathtracer") scene.integrator = PATHTRACER;
            else scene.integrator = RAYTRACER;
        }
        else if (cmd == "quadLight") {
            QuadLight q;
            ss >> q.a.x >> q.a.y >> q.a.z;
            ss >> q.ab.x >> q.ab.y >> q.ab.z;
            ss >> q.ac.x >> q.ac.y >> q.ac.z;
            ss >> q.intensity.x >> q.intensity.y >> q.intensity.z;
            scene.quadLights.push_back(q);
        }
        else if (cmd == "lightsamples") {
            ss >> scene.lightsamples;
        }
        else if (cmd == "lightstratify") {
            std::string v;
            ss >> v;
            scene.lightstratify = (v == "on");
        }
        else if (cmd == "spp") {
            ss >> scene.spp;
        }
        else if (cmd == "nexteventestimation") {
            std::string v;
            ss >> v;
            if (v == "on") scene.nexteventestimation = NEE_ON;
            else if (v == "mis") scene.nexteventestimation = NEE_MIS;
            else scene.nexteventestimation = NEE_OFF;
        }
        else if (cmd == "russianroulette") {
            std::string v;
            ss >> v;
            scene.russianroulette = (v == "on");
        }
        else if (cmd == "importancesampling") {
            std::string v;
            ss >> v;
            if (v == "cosine") scene.importancesampling = COSINE;
            else if (v == "brdf") scene.importancesampling = BRDF;
            else scene.importancesampling = HEMISPHERE;
        }
        else if (cmd == "brdf") {
            std::string v;
            ss >> v;
            curMat.brdf = (v == "ggx") ? GGX : PHONG;
        }
        else if (cmd == "roughness") {
            ss >> curMat.roughness;
        }
        else if (cmd == "gamma") {
            ss >> scene.gamma;
        }
        else if (cmd == "sigma_a") {
            ss >> scene.globalMedium.sigma_a.x >> scene.globalMedium.sigma_a.y >> scene.globalMedium.sigma_a.z;
        }
        else if (cmd == "sigma_s") {
            ss >> scene.globalMedium.sigma_s.x >> scene.globalMedium.sigma_s.y >> scene.globalMedium.sigma_s.z;
        }
        else if (cmd == "g") {
            ss >> scene.globalMedium.g;
        }
    }

    // quad lights - default material with the light's radiance as emission
    for (auto& q : scene.quadLights) {
        Material lm;
        lm.emission = q.intensity;
        q.matId = materialId(scene, lm);
    }

    infile.close();
    return scene;
}