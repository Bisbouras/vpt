// Rng.h - stateless sampler: pcg hash of seed, pixel, sample and dimension
#pragma once

#include "Common.h"
#include <cstdint>

// pcgHash - one pcg step plus its rxs-m-xs output permutation (jarzynski & olano 2020, "hash functions for gpu rendering")
HD inline uint32_t pcgHash(uint32_t v) {
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// rng - numbers for one sample of one pixel, depending only on (seed, pixel, sample, dim).
// sample s no longer depends on how many numbers earlier samples used
struct Rng {
    uint32_t key;   // hash of seed, pixel and sample
    uint32_t dim;   // next dimension

    HD Rng(uint32_t seed, uint32_t pixel, uint32_t sample)
        : key(pcgHash(sample + pcgHash(pixel + pcgHash(seed)))), dim(0) {}

    // next - uniform in [0, 1) with 24 bits. dim is spread by the golden ratio before the xor,
    // so two samples' sequences can't line up as shifted copies of each other
    HD float next() {
        uint32_t h = pcgHash(key ^ (dim++ * 0x9e3779b9u));
        return (float)(h >> 8) * (1.0f / 16777216.0f);
    }
};
