#pragma once

#include "Scene.h"
#include "SceneView.h"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <glm/gtc/constants.hpp>
#include <random>
#include <atomic>
#include <iostream>

constexpr float EPS = 1e-4f;

// beer-lambert transmittance over a path of length d through the global medium.
// componentwise rgb so colored fog works.
HD inline vec3 transmittance(float d, const SceneView& sc) {
    vec3 sigma_t = sc.globalMedium.sigma_a + sc.globalMedium.sigma_s;
    if (sigma_t == vec3(0)) return vec3(1);
    return glm::exp(-sigma_t * d);
}

// orthonormal basis aligned with w. fall back to right vector when w is near up.
HD inline void onbBasis(const vec3& w, vec3& u, vec3& v) {
    vec3 a = (fabs(w.y) > 0.99f) ? vec3(1, 0, 0) : vec3(0, 1, 0);
    u = glm::normalize(glm::cross(a, w));
    v = glm::cross(w, u);
}

// ray-sphere intersection
// pretransform the ray into object space, intersect, then transform back
HD inline Hit hitSphere(const Ray& ray, const Sphere& sph) {
    Hit res;

    vec3 roObj = vec3(sph.invTransform * vec4(ray.origin, 1.0f));
    vec3 rdObj = vec3(sph.invTransform * vec4(ray.direction, 0.0f));
    // dont normalize - keeps t consistent

    vec3 oc = roObj - sph.center;
    float a = glm::dot(rdObj, rdObj);
    float b = 2.0f * glm::dot(oc, rdObj);
    float c = glm::dot(oc, oc) - sph.radius * sph.radius;
    float disc = b * b - 4 * a * c;

    if (disc < 0) return res;

    float sq = sqrtf(disc);
    float t1 = (-b - sq) / (2 * a);
    float t2 = (-b + sq) / (2 * a);

    float t = -1;
    if (t1 > EPS) t = t1;
    else if (t2 > EPS) t = t2;
    else return res;

    vec3 hitObj = roObj + t * rdObj;
    vec3 nObj = glm::normalize(hitObj - sph.center);
    // normal transform: inverse transpose of 3x3
    vec3 nWorld = glm::normalize(vec3(glm::transpose(mat3(sph.invTransform)) * nObj));

    res.found = true;
    res.t = t;
    res.pos = ray.origin + t * ray.direction;
    res.norm = nWorld;
    res.matId = sph.matId;
    return res;
}

// moller-trumbore
HD inline Hit hitTriangle(const Ray& ray, const Triangle& tri) {
    Hit res;

    vec3 e1 = tri.v1 - tri.v0;
    vec3 e2 = tri.v2 - tri.v0;
    vec3 p = glm::cross(ray.direction, e2);
    float det = glm::dot(e1, p);

    if (fabs(det) < 1e-8f) return res;

    float inv = 1.0f / det;
    vec3 tvec = ray.origin - tri.v0;
    float u = glm::dot(tvec, p) * inv;
    if (u < 0 || u > 1) return res;

    vec3 q = glm::cross(tvec, e1);
    float v = glm::dot(ray.direction, q) * inv;
    if (v < 0 || u + v > 1) return res;

    float t = glm::dot(e2, q) * inv;
    if (t < EPS) return res;

    res.found = true;
    res.t = t;
    res.pos = ray.origin + t * ray.direction;
    res.norm = tri.normal;
    res.matId = tri.matId;
    return res;
}

// traverse BVH to find closest hit
HD inline Hit traceBVH(const Ray& ray, const SceneView& sc, float maxT = FLT_MAX) {
    Hit closest;
    closest.t = maxT;
    closest.found = false;

    if (sc.numNodes == 0) return closest;

    vec3 invd = safeInv(ray.direction);

    // iterative traversal with stack
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int ni = stack[--sp];
        const BVHNode& nd = sc.nodes[ni];

        if (!nd.box.hit(ray.origin, invd, closest.t))
            continue;

        if (nd.prim >= 0) {
            // leaf - spheres first, then tris (see BVH::reorder)
            Hit h;
            if (nd.prim < sc.numSpheres)
                h = hitSphere(ray, sc.spheres[nd.prim]);
            else
                h = hitTriangle(ray, sc.tris[nd.prim - sc.numSpheres]);

            if (h.found && h.t < closest.t && h.t > EPS)
                closest = h;
        } else {
            if (nd.left >= 0) stack[sp++] = nd.left;
            if (nd.right >= 0) stack[sp++] = nd.right;
        }
    }

    return closest;
}

HD inline bool shadow(const vec3& pt, const vec3& dir, float maxd, const SceneView& sc) {
    Ray sray(pt + EPS * dir, dir);
    Hit h = traceBVH(sray, sc, maxd);
    return h.found;
}

// arvo's analytic irradiance from a polygonal light source
// returns the irradiance vector phi; caller dots with surface normal
inline vec3 arvoPhi(const vec3& pt, const vec3 poly[], int n) {
    vec3 phi(0);
    for (int k = 0; k < n; k++) {
        vec3 a = poly[k] - pt;
        vec3 b = poly[(k + 1) % n] - pt;
        float la = glm::length(a);
        float lb = glm::length(b);
        if (la < 1e-8f || lb < 1e-8f) continue;
        float cosTheta = glm::clamp(glm::dot(a, b) / (la * lb), -1.0f, 1.0f);
        float theta = acosf(cosTheta);
        vec3 gamma = glm::normalize(glm::cross(a, b));
        phi += 0.5f * theta * gamma;
    }
    return phi;
}

// analytic direct lighting for diffuse (lambertian) surfaces
inline vec3 shadeAnalytic(const Hit& h, const SceneView& sc) {
    const Material& hm = sc.materials[h.matId];
    vec3 col = hm.emission;
    vec3 n = glm::normalize(h.norm);

    for (int qi = 0; qi < sc.numQuadLights; qi++) {
        const QuadLight& q = sc.quadLights[qi];
        // build the 4 corners in order
        vec3 poly[4] = {
            q.a,
            q.a + q.ab,
            q.a + q.ab + q.ac,
            q.a + q.ac
        };
        vec3 phi = arvoPhi(h.pos, poly, 4);
        float irr = glm::dot(phi, n);
        if (irr < 0) irr = -irr;  // handle winding
        col += (hm.diffuse / glm::pi<float>()) * q.intensity * irr;
    }
    return col;
}

// modified phong brdf
// f = kd/pi + ks * (s+2)/(2*pi) * max(0, dot(R, wo))^s
// where R is wi reflected about n
HD inline vec3 modPhongBRDF(const vec3& wi, const vec3& wo, const vec3& n,
                  const vec3& kd, const vec3& ks, float s) {
    vec3 diff = kd / glm::pi<float>();
    vec3 r = glm::reflect(-wi, n);
    float rdotwo = maxf(glm::dot(r, wo), 0.0f);
    vec3 spec = ks * ((s + 2.0f) / (2.0f * glm::pi<float>())) * powf(rdotwo, s);
    return diff + spec;
}

// schlick's approximation of fresnel reflectance
HD inline vec3 fresnelSchlick(const vec3& ks, float widoth) {
    float c = 1.0f - widoth;
    float c5 = c * c * c * c * c;
    return ks + (vec3(1.0f) - ks) * c5;
}

// ggx microfacet distribution
HD inline float ggxD(float ndoth, float alpha) {
    if (ndoth <= 0) return 0;
    float a2 = alpha * alpha;
    float c2 = ndoth * ndoth;
    float tan2 = (1.0f - c2) / c2;
    float k = a2 + tan2;
    return a2 / (glm::pi<float>() * c2 * c2 * k * k);
}

// smith g1 monodirectional shadowing-masking
HD inline float ggxG1(float vdotn, float alpha) {
    if (vdotn <= 0) return 0;
    float c2 = vdotn * vdotn;
    float tan2 = (1.0f - c2) / c2;
    return 2.0f / (1.0f + sqrtf(1.0f + alpha * alpha * tan2));
}

// ggx specular + lambertian diffuse mixture brdf
HD inline vec3 ggxBRDF(const vec3& wi, const vec3& wo, const vec3& n,
             const vec3& kd, const vec3& ks, float alpha) {
    float ndotwi = glm::dot(n, wi);
    float ndotwo = glm::dot(n, wo);
    if (ndotwi <= 0 || ndotwo <= 0) return vec3(0);
    vec3 h = glm::normalize(wi + wo);
    float ndoth = glm::dot(n, h);
    float widoth = maxf(glm::dot(wi, h), 0.0f);
    vec3 F = fresnelSchlick(ks, widoth);
    float G = ggxG1(ndotwi, alpha) * ggxG1(ndotwo, alpha);
    float D = ggxD(ndoth, alpha);
    vec3 spec = (F * G * D) / (4.0f * ndotwi * ndotwo);
    vec3 diff = kd / glm::pi<float>();
    return diff + spec;
}

// brdf dispatcher based on material type
HD inline vec3 evalBRDF(const Material& m, const vec3& wi, const vec3& wo, const vec3& n) {
    if (m.brdf == GGX) return ggxBRDF(wi, wo, n, m.diffuse, m.specular, m.roughness);
    return modPhongBRDF(wi, wo, n, m.diffuse, m.specular, m.shininess);
}

// monte carlo direct lighting with optional stratified sampling
HD inline vec3 shadeDirect(const Hit& h, const Ray& ray, const SceneView& sc,
                 std::mt19937& rng) {
    const Material& hm = sc.materials[h.matId];
    vec3 col = hm.emission;
    vec3 n = glm::normalize(h.norm);
    vec3 wo = glm::normalize(ray.origin - h.pos);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);

    for (int qi = 0; qi < sc.numQuadLights; qi++) {
        const QuadLight& q = sc.quadLights[qi];
        vec3 abxac = glm::cross(q.ab, q.ac);
        vec3 nl = glm::normalize(abxac);
        float A = glm::length(abxac);

        // orient nl so it points away from shading point (convention: nl . wi > 0)
        vec3 centerDir = (q.a + 0.5f * q.ab + 0.5f * q.ac) - h.pos;
        if (glm::dot(nl, centerDir) < 0) nl = -nl;

        int N = sc.lightsamples;
        int root = sc.lightstratify ? (int)roundf(sqrtf((float)N)) : 0;

        vec3 sum(0);
        for (int k = 0; k < N; k++) {
            // sample point on quad
            float u1, u2;
            if (sc.lightstratify) {
                int j = k / root;
                int i = k % root;
                u1 = (j + U(rng)) / root;
                u2 = (i + U(rng)) / root;
            } else {
                u1 = U(rng);
                u2 = U(rng);
            }
            vec3 xp = q.a + u1 * q.ab + u2 * q.ac;

            vec3 toL = xp - h.pos;
            float R = glm::length(toL);
            vec3 wi = toL / R;

            // visibility
            if (shadow(h.pos, wi, R - EPS, sc)) continue;

            float ndotwi = maxf(glm::dot(n, wi), 0.0f);
            float nldotwi = maxf(glm::dot(nl, wi), 0.0f);  // nl already oriented away from scene

            if (ndotwi <= 0 || nldotwi <= 0) continue;

            vec3 f = evalBRDF(hm, wi, wo, n);
            float G = (ndotwi * nldotwi) / (R * R);
            sum += f * G * transmittance(R, sc);
        }
        col += q.intensity * (A / (float)N) * sum;
    }
    return col;
}

inline vec3 shade(const Hit& h, const Ray& ray, const SceneView& sc) {
    const Material& hm = sc.materials[h.matId];
    vec3 col = hm.ambient + hm.emission;
    vec3 n = glm::normalize(h.norm);
    vec3 v = glm::normalize(ray.origin - h.pos);

    // directional lights
    for (int li = 0; li < sc.numDirLights; li++) {
        const DirLight& l = sc.dirLights[li];
        vec3 ld = glm::normalize(l.dir);
        if (shadow(h.pos, ld, FLT_MAX, sc)) continue;

        float ndl = std::max(glm::dot(n, ld), 0.0f);
        col += hm.diffuse * l.col * ndl;

        // blinn-phong specular
        vec3 half = glm::normalize(ld + v);
        float ndh = std::max(glm::dot(n, half), 0.0f);
        if (ndl > 0)
            col += hm.specular * l.col * powf(ndh, hm.shininess);
    }

    // point lights
    for (int li = 0; li < sc.numPtLights; li++) {
        const PointLight& l = sc.ptLights[li];
        vec3 toL = l.pos - h.pos;
        float dist = glm::length(toL);
        vec3 ld = glm::normalize(toL);

        if (shadow(h.pos, ld, dist, sc)) continue;

        // attenuation (default 1,0,0 = no falloff)
        float att = 1.0f / (sc.atten.x + sc.atten.y * dist + sc.atten.z * dist * dist);

        float ndl = std::max(glm::dot(n, ld), 0.0f);
        col += hm.diffuse * l.col * ndl * att;

        vec3 half = glm::normalize(ld + v);
        float ndh = std::max(glm::dot(n, half), 0.0f);
        if (ndl > 0)
            col += hm.specular * l.col * powf(ndh, hm.shininess) * att;
    }

    return col;
}

// recursive raytrace with mirror reflection
inline vec3 trace(const Ray& ray, const SceneView& sc, int depth) {
    if (depth <= 0) return vec3(0);

    Hit h = traceBVH(ray, sc);
    if (!h.found) return vec3(0);
    const Material& hm = sc.materials[h.matId];

    vec3 col = shade(h, ray, sc);

    // reflection
    if (hm.specular != vec3(0) && depth > 0) {
        vec3 rdir = glm::reflect(ray.direction, glm::normalize(h.norm));
        Ray rray(h.pos + EPS * rdir, rdir);
        col += hm.specular * trace(rray, sc, depth - 1);
    }

    return col;
}

HD inline Ray makeRay(float px, float py, const SceneView& sc) {
    float aspect = (float)sc.w / (float)sc.h;
    float fovyr = glm::radians(sc.fovy);
    float fovxr = 2.0f * atanf(tanf(fovyr / 2.0f) * aspect);

    vec3 w = glm::normalize(sc.eye - sc.center);
    vec3 u = glm::normalize(glm::cross(sc.up, w));
    vec3 v = glm::cross(w, u);

    float a = tanf(fovxr / 2.0f) * ((px - sc.w / 2.0f) / (sc.w / 2.0f));
    float b = tanf(fovyr / 2.0f) * ((sc.h / 2.0f - py) / (sc.h / 2.0f));

    return Ray(sc.eye, glm::normalize(a * u + b * v - w));
}

HD inline bool hitQuad(const Ray& ray, const QuadLight& q, float& tOut) {
    vec3 nl = glm::cross(q.ab, q.ac);
    float denom = glm::dot(ray.direction, nl);
    if (fabs(denom) < 1e-8f) return false;

    float t = glm::dot(q.a - ray.origin, nl) / denom;
    if (t < EPS) return false;

    vec3 p = ray.origin + t * ray.direction - q.a;
    float u = glm::dot(p, q.ab) / glm::dot(q.ab, q.ab);
    float v = glm::dot(p, q.ac) / glm::dot(q.ac, q.ac);
    if (u < 0 || u > 1 || v < 0 || v > 1) return false;

    tOut = t;
    return true;
}

// like traceBVH but also intersects quad lights, synthesizing a Hit with
// emission set. for primary + indirect rays in the path tracer.
// shadow rays still use traceBVH (the light shouldn't shadow itself).
HD inline Hit traceScene(const Ray& ray, const SceneView& sc, float maxT = FLT_MAX) {
    Hit closest = traceBVH(ray, sc, maxT);

    for (int k = 0; k < sc.numQuadLights; k++) {
        float t;
        if (hitQuad(ray, sc.quadLights[k], t) && t < closest.t && t > EPS) {
            closest.found = true;
            closest.t = t;
            closest.pos = ray.origin + t * ray.direction;
            vec3 nl = glm::normalize(glm::cross(sc.quadLights[k].ab, sc.quadLights[k].ac));
            if (glm::dot(nl, ray.direction) > 0) nl = -nl;
            closest.norm = nl;
            closest.matId = sc.quadLights[k].matId;
        }
    }

    return closest;
}

// reflectiveness t = avg(ks) / (avg(kd) + avg(ks)). returns 1 if both zero.
HD inline float phongT(const vec3& kd, const vec3& ks) {
    float ad = (kd.r + kd.g + kd.b) / 3.0f;
    float as = (ks.r + ks.g + ks.b) / 3.0f;
    float d = ad + as;
    if (d <= 1e-8f) return 1.0f;
    return as / d;
}

// ggx reflectiveness: same as phong but with a 0.25 floor so fresnel always gets samples.
HD inline float ggxT(const vec3& kd, const vec3& ks) {
    float ad = (kd.r + kd.g + kd.b) / 3.0f;
    float as = (ks.r + ks.g + ks.b) / 3.0f;
    float d = ad + as;
    if (d <= 1e-8f) return 1.0f;
    return maxf(0.25f, as / d);
}

// sample wi from the modified phong importance sampling pdf.
// xi0 picks diffuse vs specular lobe; (xi1, xi2) generate the spherical coords.
HD inline vec3 samplePhongDir(const Material& m, const vec3& wo, const vec3& n,
                    float xi0, float xi1, float xi2) {
    float t = phongT(m.diffuse, m.specular);
    float phi = 2.0f * glm::pi<float>() * xi2;
    float theta;
    vec3 axis;
    if (xi0 <= t) {
        // specular lobe centered on the reflection of wo about n
        theta = acosf(powf(xi1, 1.0f / (m.shininess + 1.0f)));
        axis = glm::reflect(-wo, n);
    } else {
        // diffuse cosine lobe centered on the normal
        theta = acosf(sqrtf(xi1));
        axis = n;
    }
    vec3 s(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
    vec3 u, v;
    onbBasis(axis, u, v);
    return glm::normalize(s.x * u + s.y * v + s.z * axis);
}

// evaluate the modified phong importance sampling pdf at arbitrary wi
HD inline float pdfPhongDir(const Material& m, const vec3& wi, const vec3& wo, const vec3& n) {
    float t = phongT(m.diffuse, m.specular);
    float ndotwi = maxf(glm::dot(n, wi), 0.0f);
    vec3 r = glm::reflect(-wo, n);
    float rdotwi = maxf(glm::dot(r, wi), 0.0f);
    float pd = ndotwi / glm::pi<float>();
    float ps = ((m.shininess + 1.0f) / (2.0f * glm::pi<float>())) * powf(rdotwi, m.shininess);
    return (1.0f - t) * pd + t * ps;
}

// sample wi from the ggx importance sampling pdf.
// specular lobe samples a half vector from D then reflects wo about it.
HD inline vec3 sampleGGXDir(const Material& m, const vec3& wo, const vec3& n,
                  float xi0, float xi1, float xi2) {
    float t = ggxT(m.diffuse, m.specular);
    float phi = 2.0f * glm::pi<float>() * xi2;
    if (xi0 <= t) {
        float alpha = m.roughness;
        float theta_h = atanf(alpha * sqrtf(xi1) / sqrtf(1.0f - xi1));
        vec3 hLocal(cosf(phi) * sinf(theta_h), sinf(phi) * sinf(theta_h), cosf(theta_h));
        vec3 u, v;
        onbBasis(n, u, v);
        vec3 h = glm::normalize(hLocal.x * u + hLocal.y * v + hLocal.z * n);
        return glm::normalize(2.0f * glm::dot(wo, h) * h - wo);
    } else {
        float theta = acosf(sqrtf(xi1));
        vec3 s(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
        vec3 u, v;
        onbBasis(n, u, v);
        return glm::normalize(s.x * u + s.y * v + s.z * n);
    }
}

// evaluate the ggx importance sampling pdf at arbitrary wi
HD inline float pdfGGXDir(const Material& m, const vec3& wi, const vec3& wo, const vec3& n) {
    float t = ggxT(m.diffuse, m.specular);
    float ndotwi = maxf(glm::dot(n, wi), 0.0f);
    vec3 h = glm::normalize(wi + wo);
    float ndoth = maxf(glm::dot(n, h), 0.0f);
    float hdotwi = maxf(glm::dot(h, wi), 1e-8f);
    float pd = ndotwi / glm::pi<float>();
    float D = ggxD(ndoth, m.roughness);
    float ps = D * ndoth / (4.0f * hdotwi);
    return (1.0f - t) * pd + t * ps;
}

// brdf sampling dispatcher
HD inline vec3 sampleBRDFDir(const Material& m, const vec3& wo, const vec3& n,
                   float xi0, float xi1, float xi2) {
    if (m.brdf == GGX) return sampleGGXDir(m, wo, n, xi0, xi1, xi2);
    return samplePhongDir(m, wo, n, xi0, xi1, xi2);
}

HD inline float pdfBRDFDir(const Material& m, const vec3& wi, const vec3& wo, const vec3& n) {
    if (m.brdf == GGX) return pdfGGXDir(m, wi, wo, n);
    return pdfPhongDir(m, wi, wo, n);
}

// per-light angular-measure pdf: R^2 / (A * |nl . wi|) if the ray hits light k, else 0
HD inline float pdfLightK(const vec3& pos, const vec3& wi, const QuadLight& q) {
    Ray r(pos + EPS * wi, wi);
    float t;
    if (!hitQuad(r, q, t)) return 0;
    vec3 abxac = glm::cross(q.ab, q.ac);
    float A = glm::length(abxac);
    if (A < 1e-8f) return 0;
    vec3 nl = abxac / A;
    float nldotwi = fabs(glm::dot(nl, wi));
    if (nldotwi < 1e-8f) return 0;
    return (t * t) / (A * nldotwi);
}

// nee pdf averaged over all quad lights
HD inline float pdfNEE(const vec3& pos, const vec3& wi, const SceneView& sc) {
    int n = sc.numQuadLights;
    if (n == 0) return 0;
    float sum = 0;
    for (int qi = 0; qi < sc.numQuadLights; qi++) sum += pdfLightK(pos, wi, sc.quadLights[qi]);
    return sum / (float)n;
}

// multi-sample mis direct lighting with power heuristic (beta = 2).
// uses one nee sample per light plus one brdf sample. caller adds the hit material's emission.
HD inline vec3 shadeMIS(const Hit& h, const Ray& ray, const SceneView& sc,
              std::mt19937& rng) {
    const Material& hm = sc.materials[h.matId];
    vec3 col(0);
    vec3 n = glm::normalize(h.norm);
    vec3 wo = glm::normalize(ray.origin - h.pos);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    int nlights = sc.numQuadLights;

    // nee technique: one sample per light, weighted by w_nee
    for (int k = 0; k < nlights; k++) {
        const QuadLight& q = sc.quadLights[k];
        vec3 abxac = glm::cross(q.ab, q.ac);
        float A = glm::length(abxac);
        if (A < 1e-8f) continue;
        vec3 nl = abxac / A;

        float u1 = U(rng), u2 = U(rng);
        vec3 xp = q.a + u1 * q.ab + u2 * q.ac;
        vec3 toL = xp - h.pos;
        float R = glm::length(toL);
        vec3 wi = toL / R;

        float ndotwi = glm::dot(n, wi);
        float nldotwi = fabs(glm::dot(nl, wi));
        if (ndotwi <= 0 || nldotwi < 1e-8f) continue;
        if (shadow(h.pos, wi, R - EPS, sc)) continue;

        float pdfNee = pdfNEE(h.pos, wi, sc);
        if (pdfNee <= 0) continue;
        float pdfBrdf = pdfBRDFDir(hm, wi, wo, n);

        // power heuristic with sample counts: w_i = (N_i pdf_i)^2 / sum (N_k pdf_k)^2
        // N_nee = nlights, N_brdf = 1
        float wn = (float)nlights * pdfNee;
        float w = (wn * wn) / (wn * wn + pdfBrdf * pdfBrdf);
        vec3 f = evalBRDF(hm, wi, wo, n);
        // (1/N_nee) * w * f * Li * (n.wi) / pdf_nee, with N_nee = nlights
        col += (1.0f / (float)nlights) * w * f * q.intensity * ndotwi * transmittance(R, sc) / pdfNee;
    }

    // brdf technique: one sample, weighted by w_brdf
    float xi0 = U(rng), xi1 = U(rng), xi2 = U(rng);
    vec3 wi = sampleBRDFDir(hm, wo, n, xi0, xi1, xi2);
    float ndotwi = glm::dot(n, wi);
    if (ndotwi > 0) {
        float pdfBrdf = pdfBRDFDir(hm, wi, wo, n);
        if (pdfBrdf > 1e-8f) {
            // does the brdf ray reach a light unobstructed?
            Ray sr(h.pos + EPS * wi, wi);
            Hit geomHit = traceBVH(sr, sc);
            float geomT = geomHit.found ? geomHit.t : FLT_MAX;
            int qhit = -1;
            float qt = geomT;
            for (int k = 0; k < nlights; k++) {
                float qtk;
                if (hitQuad(sr, sc.quadLights[k], qtk) && qtk < qt) {
                    qt = qtk;
                    qhit = k;
                }
            }
            if (qhit >= 0) {
                float pdfNee = pdfNEE(h.pos, wi, sc);
                float wn = (float)nlights * pdfNee;
                float w = (pdfBrdf * pdfBrdf) / (pdfBrdf * pdfBrdf + wn * wn);
                vec3 f = evalBRDF(hm, wi, wo, n);
                col += w * f * sc.quadLights[qhit].intensity * ndotwi * transmittance(qt, sc) / pdfBrdf;
            }
        }
    }

    return col;
}

// henyey-greenstein phase function. p = (1-g^2) / (4pi * (1 + g^2 - 2g*cos)^(3/2))
HD inline float hgPhase(float cosTheta, float g) {
    if (fabsf(g) < 1e-3f) return 1.0f / (4.0f * glm::pi<float>());
    float denom = 1.0f + g*g - 2.0f * g * cosTheta;
    return (1.0f - g*g) / (4.0f * glm::pi<float>() * denom * sqrtf(denom));
}

// sample a new direction from HG given the incoming direction wo (points back to last vertex).
// cosTheta is the angle between the new direction and the ray's forward direction (-wo).
// returns the new direction and sets pdf to the HG value at that direction.
HD inline vec3 sampleHG(const vec3& wo, float g, std::mt19937& rng, float& pdf) {
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    float xi1 = U(rng), xi2 = U(rng);
    float cosTheta;
    if (fabsf(g) < 1e-3f) {
        cosTheta = 1.0f - 2.0f * xi1;
    } else {
        // inverse cdf on cosTheta for HG
        float sq = (1.0f - g*g) / (1.0f - g + 2.0f * g * xi1);
        cosTheta = (1.0f + g*g - sq*sq) / (2.0f * g);
    }
    cosTheta = glm::clamp(cosTheta, -1.0f, 1.0f);
    float sinTheta = sqrtf(glm::max(1.0f - cosTheta * cosTheta, 0.0f));
    float phi = 2.0f * glm::pi<float>() * xi2;

    // basis around the ray's forward direction
    vec3 fwd = -wo;
    vec3 u, v;
    onbBasis(fwd, u, v);
    vec3 wi = sinTheta * cosf(phi) * u + sinTheta * sinf(phi) * v + cosTheta * fwd;
    pdf = hgPhase(cosTheta, g);
    return glm::normalize(wi);
}

// mis direct lighting at a medium scattering vertex.
// same structure as shadeMIS but with the phase function in place of the BRDF and
// no surface normal / cosine factor.
HD inline vec3 shadeMISVolume(const vec3& p, const vec3& wo, const SceneView& sc,
                    std::mt19937& rng) {
    vec3 col(0);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    int nlights = sc.numQuadLights;
    float g = sc.globalMedium.g;

    // nee: one sample per light
    for (int k = 0; k < nlights; k++) {
        const QuadLight& q = sc.quadLights[k];
        vec3 abxac = glm::cross(q.ab, q.ac);
        float A = glm::length(abxac);
        if (A < 1e-8f) continue;
        vec3 nl = abxac / A;

        float u1 = U(rng), u2 = U(rng);
        vec3 xp = q.a + u1 * q.ab + u2 * q.ac;
        vec3 toL = xp - p;
        float R = glm::length(toL);
        vec3 wi = toL / R;

        float nldotwi = fabs(glm::dot(nl, wi));
        if (nldotwi < 1e-8f) continue;
        if (shadow(p, wi, R - EPS, sc)) continue;

        float pdfNee = pdfNEE(p, wi, sc);
        if (pdfNee <= 0) continue;
        // phase pdf in solid angle. cosTheta between wi and forward dir (-wo)
        float cosTheta = -glm::dot(wo, wi);
        float pdfPh = hgPhase(cosTheta, g);

        // power heuristic with sample counts. N_nee = nlights, N_phase = 1
        float wn = (float)nlights * pdfNee;
        float w = (wn * wn) / (wn * wn + pdfPh * pdfPh);
        // (1/N_nee) * w * p_HG * Li * transmittance(R) / pdf_nee, no cosine at volume vertex
        col += (1.0f / (float)nlights) * w * pdfPh * q.intensity * transmittance(R, sc) / pdfNee;
    }

    // phase function technique
    float pdfPh;
    vec3 wi = sampleHG(wo, g, rng, pdfPh);
    if (pdfPh > 1e-8f) {
        Ray sr(p + EPS * wi, wi);
        Hit geomHit = traceBVH(sr, sc);
        float geomT = geomHit.found ? geomHit.t : FLT_MAX;
        int qhit = -1;
        float qt = geomT;
        for (int k = 0; k < nlights; k++) {
            float qtk;
            if (hitQuad(sr, sc.quadLights[k], qtk) && qtk < qt) {
                qt = qtk;
                qhit = k;
            }
        }
        if (qhit >= 0) {
            float pdfNee = pdfNEE(p, wi, sc);
            float wn = (float)nlights * pdfNee;
            float w = (pdfPh * pdfPh) / (pdfPh * pdfPh + wn * wn);
            // phase-IS contribution simplifies: p_HG/pdfPh = 1
            col += w * sc.quadLights[qhit].intensity * transmittance(qt, sc);
        }
    }

    return col;
}

// iterative path tracer with optional NEE/MIS, brdf importance sampling, and russian roulette.
// maxdepth < 0 means unbounded (only RR can terminate).
HD inline vec3 pathTrace(const Ray& primary, const SceneView& sc, std::mt19937& rng) {
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    vec3 L(0);
    vec3 T(1);
    Ray ray = primary;
    bool infinite = sc.maxdepth < 0;
    bool useNEE = (sc.nexteventestimation == NEE_ON);
    bool useMIS = (sc.nexteventestimation == NEE_MIS);
    bool anyDirect = useNEE || useMIS;

    // medium state. hasScattering picks the free-flight path; otherwise pure absorption.
    vec3 sigma_t = sc.globalMedium.sigma_a + sc.globalMedium.sigma_s;
    vec3 sigma_s = sc.globalMedium.sigma_s;
    bool hasScattering = (sigma_s != vec3(0));
    // scalar sigma_t for sampling (mean of channels). spectral correction handles the difference.
    float sigma_t_s = (sigma_t.r + sigma_t.g + sigma_t.b) / 3.0f;

    for (int d = 0; infinite || d < sc.maxdepth; d++) {
        Hit h = traceScene(ray, sc);
        if (!h.found) break;
        const Material& hm = sc.materials[h.matId];

        if (hasScattering) {
            // free-flight sampling. tSample ~ exp(sigma_t_s)
            float xi = U(rng);
            float tSample = -logf(1.0f - xi) / sigma_t_s;
            if (tSample < h.t) {
                // scatter event inside the medium at scatterPos
                vec3 scatterPos = ray.origin + tSample * ray.direction;
                vec3 woVol = -ray.direction;

                // throughput update at the scatter vertex.
                //   T *= sigma_s * exp(-sigma_t*t) / (sigma_t_s * exp(-sigma_t_s*t))
                //   = (sigma_s / sigma_t_s) * exp((sigma_t_s - sigma_t) * t)
                T *= (sigma_s / sigma_t_s) * glm::exp((sigma_t_s - sigma_t) * tSample);

                // direct light estimator at the scatter vertex
                if (anyDirect) {
                    L += T * shadeMISVolume(scatterPos, woVol, sc, rng);
                }

                bool lastHop = !infinite && (d == sc.maxdepth - 1);
                if (lastHop) break;

                if (sc.russianroulette) {
                    float maxC = glm::max(glm::max(T.r, T.g), T.b);
                    float q = 1.0f - glm::min(maxC, 1.0f);
                    if (U(rng) < q) break;
                    T /= (1.0f - q);
                }

                // sample HG for next direction. pdf cancels in throughput (perfect IS).
                float pdfPh;
                vec3 wi = sampleHG(woVol, sc.globalMedium.g, rng, pdfPh);
                if (pdfPh < 1e-8f) break;

                ray = Ray(scatterPos + EPS * wi, wi);
                continue;
            } else {
                // no scatter in [0, h.t]. spectral correction for sampling with mean sigma_t.
                //   T *= exp(-sigma_t*h.t) / exp(-sigma_t_s*h.t) = exp((sigma_t_s - sigma_t) * h.t)
                T *= glm::exp((sigma_t_s - sigma_t) * h.t);
            }
        } else if (sigma_t != vec3(0)) {
            // absorption-only fallback (milestone behavior). beer-lambert along the segment.
            T *= transmittance(h.t, sc);
        }

        vec3 n = glm::normalize(h.norm);
        vec3 wo = glm::normalize(ray.origin - h.pos);

        // emission accounting
        if (d == 0) {
            L += T * hm.emission;
            if (hm.emission != vec3(0) && anyDirect) break;
        } else if (hm.emission != vec3(0)) {
            // for NEE/MIS the previous bounce's direct estimator already accounted for this
            if (anyDirect) break;
            L += T * hm.emission;
            break;
        }

        // direct light estimator at this vertex
        if (useNEE) {
            // shadeDirect adds hm.emission as col init, so subtract it here
            L += T * (shadeDirect(h, ray, sc, rng) - hm.emission);
        } else if (useMIS) {
            L += T * shadeMIS(h, ray, sc, rng);
        }

        // last bounce: stop before sampling next direction
        bool lastHop = !infinite && (d == sc.maxdepth - 1);
        if (lastHop) break;

        // russian roulette (after direct, before indirect sample)
        // direct used unboosted T - the boost only applies to the next bounce
        if (sc.russianroulette) {
            float maxC = glm::max(glm::max(T.r, T.g), T.b);
            float q = 1.0f - glm::min(maxC, 1.0f);
            if (U(rng) < q) break;
            T /= (1.0f - q);
        }

        // sample next direction
        vec3 wi;
        float pdfval;
        if (sc.importancesampling == BRDF) {
            float xi0 = U(rng), xi1 = U(rng), xi2 = U(rng);
            wi = sampleBRDFDir(hm, wo, n, xi0, xi1, xi2);
            pdfval = pdfBRDFDir(hm, wi, wo, n);
        } else {
            float xi1 = U(rng), xi2 = U(rng);
            float theta = (sc.importancesampling == COSINE) ? acosf(sqrtf(xi1)) : acosf(xi1);
            float phi = 2.0f * glm::pi<float>() * xi2;
            vec3 s(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
            vec3 u, v;
            onbBasis(n, u, v);
            wi = glm::normalize(s.x * u + s.y * v + s.z * n);
            pdfval = (sc.importancesampling == COSINE)
                ? glm::max(glm::dot(n, wi), 0.0f) / glm::pi<float>()
                : 1.0f / (2.0f * glm::pi<float>());
        }

        float ndotwi = glm::dot(n, wi);
        if (ndotwi <= 0 || pdfval <= 1e-8f) break;

        // throughput update.
        // hemisphere: T *= 2pi * f * (n.wi)  (uniform pdf = 1/(2pi))
        // cosine:     T *= pi * f            (cosine pdf = (n.wi)/pi cancels the dot product)
        // brdf:       T *= f * (n.wi) / pdf  (general form)
        vec3 f = evalBRDF(hm, wi, wo, n);
        if (sc.importancesampling == COSINE)
            T *= glm::pi<float>() * f;
        else if (sc.importancesampling == HEMISPHERE)
            T *= 2.0f * glm::pi<float>() * f * ndotwi;
        else
            T *= f * ndotwi / pdfval;

        ray = Ray(h.pos + EPS * wi, wi);
    }

    return L;
}

// renderPixel - radiance of sample s of pixel (x, y). sample 0 goes through the pixel centre
HD inline vec3 renderPixel(const SceneView& sc, int x, int y, int s, std::mt19937& rng) {
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    float dx = (s == 0) ? 0.5f : U(rng);
    float dy = (s == 0) ? 0.5f : U(rng);
    Ray pr = makeRay(x + dx, y + dy, sc);
    return pathTrace(pr, sc, rng);
}

// adds each pixel's samples into sum (zeroed by the caller), returns the samples per pixel
// (spp for the path tracer, 1 for the other integrators). outputPass turns sum into images
inline int render(const Scene& sc, const BVH& bvh, vec3* sum, bool quiet) {
    const SceneView v = makeView(sc, bvh);
    int total = sc.w * sc.h;
    std::atomic<int> done(0);
    // seed - per-pixel seeds of different seed values never overlap
    unsigned seedBase = sc.seed * (unsigned)total;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < sc.h; j++) {
        for (int i = 0; i < sc.w; i++) {
            int p = j * sc.w + i;

            if (sc.integrator == PATHTRACER) {
                std::mt19937 rng(seedBase + (unsigned)p);
                for (int s = 0; s < sc.spp; s++)
                    sum[p] += renderPixel(v, i, j, s, rng);
            }
            else if (sc.integrator == RAYTRACER) {
                Ray ray = makeRay(i + 0.5f, j + 0.5f, v);
                sum[p] = trace(ray, v, sc.maxdepth);
            } else {
                Ray ray = makeRay(i + 0.5f, j + 0.5f, v);
                vec3 col(0);
                // check quad light visibility (primary ray only)
                float tq = FLT_MAX;
                int qhit = -1;
                for (size_t k = 0; k < sc.quadLights.size(); k++) {
                    float t;
                    if (hitQuad(ray, sc.quadLights[k], t) && t < tq) {
                        tq = t;
                        qhit = (int)k;
                    }
                }

                Hit h = traceBVH(ray, v);

                if (qhit >= 0 && (!h.found || tq < h.t)) {
                    col = sc.quadLights[qhit].intensity;
                } else if (h.found) {
                    if (sc.integrator == ANALYTICDIRECT) {
                        col = shadeAnalytic(h, v);
                    } else {
                        std::mt19937 rng(seedBase + (unsigned)p);
                        col = shadeDirect(h, ray, v, rng);
                    }
                }
                sum[p] = col;
            }

            int d = ++done;
            if (quiet) continue;
            int pct = (int)((d * 100LL) / total);
            if (pct % 10 == 0 && d % sc.w == 0)
                std::cout << "\r" << pct << "%" << std::flush;
        }
    }
    if (!quiet) std::cout << "\r100%" << std::endl;
    return sc.integrator == PATHTRACER ? sc.spp : 1;
}