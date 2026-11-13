#pragma once

#include "Scene.h"
#include <algorithm>
#include <numeric>
#include <cfloat>
#include <cmath>

// safeInv - 1/x with |x| < 1e-20 nudged to +-1e-20, so the result is finite and AABB::hit can't do 0 * inf (nan)
HD inline float safeInv(float x) {
    return 1.0f / (fabsf(x) > 1e-20f ? x : copysignf(1e-20f, x));
}

HD inline vec3 safeInv(const vec3& d) {
    return vec3(safeInv(d.x), safeInv(d.y), safeInv(d.z));
}

struct AABB {
    vec3 bmin = vec3(FLT_MAX);
    vec3 bmax = vec3(-FLT_MAX);

    void grow(const vec3& p) {
        bmin = glm::min(bmin, p);
        bmax = glm::max(bmax, p);
    }

    void grow(const AABB& other) {
        bmin = glm::min(bmin, other.bmin);
        bmax = glm::max(bmax, other.bmax);
    }

    vec3 centroid() const { return (bmin + bmax) * 0.5f; }

    // inv_d from safeInv - finite, so no nan when the origin lies on a slab plane
    HD bool hit(const vec3& orig, const vec3& inv_d, float tmax) const {
        vec3 t0 = (bmin - orig) * inv_d;
        vec3 t1 = (bmax - orig) * inv_d;
        vec3 lo = glm::min(t0, t1);
        vec3 hi = glm::max(t0, t1);
        float tenter = glm::max(glm::max(lo.x, lo.y), lo.z);
        float texit = glm::min(glm::min(hi.x, hi.y), hi.z);
        return tenter <= texit && texit > 0 && tenter < tmax;
    }
};

// wraps either a sphere or triangle with its bounding box (build only)
struct Prim {
    enum Kind { SPH, TRI };
    Kind kind;
    int idx;
    AABB box;
};

inline AABB getSphereBox(const Sphere& s) {
    // transform 8 corners of the local bbox to world space
    vec3 lo = s.center - vec3(s.radius);
    vec3 hi = s.center + vec3(s.radius);
    AABB b;
    for (int i = 0; i < 8; i++) {
        vec3 corner(
            (i & 1) ? hi.x : lo.x,
            (i & 2) ? hi.y : lo.y,
            (i & 4) ? hi.z : lo.z
        );
        b.grow(vec3(s.transform * vec4(corner, 1.0f)));
    }
    b.bmin -= vec3(1e-4f);
    b.bmax += vec3(1e-4f);
    return b;
}

inline AABB getTriBox(const Triangle& t) {
    AABB b;
    b.grow(t.v0);
    b.grow(t.v1);
    b.grow(t.v2);
    b.bmin -= vec3(1e-4f);
    b.bmax += vec3(1e-4f);
    return b;
}

// leaf if prim >= 0. after reorder(), prim < numSpheres is spheres[prim], the rest are tris[prim - numSpheres]
struct BVHNode {
    AABB box;
    int left = -1;
    int right = -1;
    int prim = -1;
};

struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<Prim> prims;

    void build(const Scene& sc) {
        for (int i = 0; i < (int)sc.spheres.size(); i++) {
            Prim p;
            p.kind = Prim::SPH;
            p.idx = i;
            p.box = getSphereBox(sc.spheres[i]);
            prims.push_back(p);
        }
        for (int i = 0; i < (int)sc.tris.size(); i++) {
            Prim p;
            p.kind = Prim::TRI;
            p.idx = i;
            p.box = getTriBox(sc.tris[i]);
            prims.push_back(p);
        }
        if (prims.empty()) return;

        std::vector<int> ids(prims.size());
        std::iota(ids.begin(), ids.end(), 0);
        nodes.reserve(prims.size() * 2);
        buildRec(ids, 0, (int)ids.size());
    }

    int buildRec(std::vector<int>& ids, int lo, int hi) {
        BVHNode nd;
        for (int i = lo; i < hi; i++)
            nd.box.grow(prims[ids[i]].box);

        int n = hi - lo;
        if (n == 1) {
            nd.prim = ids[lo];
            int i = (int)nodes.size();
            nodes.push_back(nd);
            return i;
        }

        // split on longest axis
        vec3 ext = nd.box.bmax - nd.box.bmin;
        int axis = 0;
        if (ext.y > ext.x) axis = 1;
        if (ext.z > ext[axis]) axis = 2;

        std::sort(ids.begin() + lo, ids.begin() + hi,
            [&](int a, int b) {
                return prims[a].box.centroid()[axis] < prims[b].box.centroid()[axis];
            });

        int mid = lo + n / 2;
        int i = (int)nodes.size();
        nodes.push_back(nd);

        int l = buildRec(ids, lo, mid);
        int r = buildRec(ids, mid, hi);
        nodes[i].left = l;
        nodes[i].right = r;
        return i;
    }

    // reorder - spheres and tris into leaf order so leaves index them directly, without Prim
    void reorder(Scene& sc) {
        int ns = (int)sc.spheres.size();
        std::vector<Sphere> sph;
        std::vector<Triangle> tri;
        sph.reserve(sc.spheres.size());
        tri.reserve(sc.tris.size());
        for (auto& nd : nodes) {
            if (nd.prim < 0) continue;
            const Prim& p = prims[nd.prim];
            if (p.kind == Prim::SPH) {
                nd.prim = (int)sph.size();
                sph.push_back(sc.spheres[p.idx]);
            }
            else {
                nd.prim = ns + (int)tri.size();
                tri.push_back(sc.tris[p.idx]);
            }
        }
        sc.spheres.swap(sph);
        sc.tris.swap(tri);
    }
};
