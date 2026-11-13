# VPT CUDA port — progress log

What happened, in order: what was done, bugs found, where the plan changed and why, and what was measured. `PLAN.md` says what we intend to do. This file records what actually happened. Phase 4 measurements go in `PERF_LOG.md`.

Entry tags:
- **Bug**: a defect in the renderer, found or fixed
- **Harness**: a problem in the test or benchmark tooling
- **Plan change**: a departure from `PLAN.md` or the original outline, with the reason
- **Finding**: something learned about the code, data or hardware that affects later work
- **Environment**: toolchain or machine
- **Result**: a measured outcome

---

## Open items

| Item | Since | Needed by |
|---|---|---|
| `ncu` can't read GPU performance counters (`ERR_NVGPUCTRPERM`). Enable "Allow access to all users" under NVIDIA Control Panel → Developer → Manage GPU Performance Counters, or run `ncu` as admin. | 2026-11-09 | Phase 4 |
| `out/build/x64-Release` (the VS build folder) still has a CMake cache from the project's old location, so it can't configure from here. The scripts use `out/build/cli-release`. Delete that folder's `CMakeCache.txt` and `CMakeFiles/` (save its old renders first if wanted), then rebuild in VS. | 2026-11-09 | Before VS builds are used |
| `hero`'s statistical check only catches errors of about 5% or more (fireflies survive in its reference). Its code paths are covered more tightly by `cornellScatter`. Since the PCG floor (2.8), `cornellCosine` catches about 2% overall and 10% in one quarter (was 1% and 5%). | 2026-11-09 | Accepted limit |
| CPU timings for `cornellScatter` and `hero` vary up to 10% between runs (P/E cores). Medians of 5 are used, and min/max are recorded. | 2026-11-09 | Keep in mind when reading small CPU changes |

## Bug register

| ID | Bug | Found | Status |
|---|---|---|---|
| B1 | `done++` in `render()` is a data race under OpenMP. It also printed progress lines from several threads. | Outline (known) | **Fixed** in Phase 1 (`ae82772`): `std::atomic<int>`, printing off with `--quiet` |
| B2 | `AABB::hit` can produce NaN for axis-aligned rays (0 × inf). CPU and GPU min/max treat NaN differently. | Outline (known) | **Fixed** in 2.6 (`safeInv`). Never triggered in the 8 scenes (0 in about 3.5 billion box tests), but shown to flip hit/miss in a constructed case |
| B3 | Functions defined in headers aren't `inline`. Linking will fail with duplicate symbols as soon as a second file (`Kernels.cu`) includes them. Latent today. | Phase 0.5 code read | **Fixed** in 2.1 |
| B4 | No bounce cap for `maxdepth -1`: only Russian roulette ends a path. Fine on the CPU, but on the GPU it means long-tailed warps and a risk of hitting the Windows 2 s kernel timeout. | Phase 0.5 code read | **Fixed** in 2.9 (`MAX_BOUNCES = 256`). Longest path seen: 100 bounces in 172 million |

## Plan changes

| ID | Change | Why | When |
|---|---|---|---|
| C1 | Phase 0.5 added: a written plan (`PLAN.md`) and toolchain check before any code | Settle design questions and catch toolchain problems up front | 2026-11-08 |
| C2 | Phase 2 order: pure refactors first, RNG swap last (D2) | Until the RNG changes, each refactor can be checked byte for byte, a much stronger test than statistics | 2026-11-08 |
| C3 | Shared bounce cap of 256 added to Phase 2 (D5) | B4 | 2026-11-08 |
| C4 | Bench spp is 128 for every scene (plan had `hero` 64–256 and `mis` 32) | Measured CPU runs at 128 spp take 4–13 s: long enough to time, and one spp keeps the table simple | 2026-11-09 |
| C5 | References rendered with their own seed (1000) | With seed 0, a 64-spp test would reuse the reference's first 64 samples per pixel and look better than it is | 2026-11-09 |
| C6 | Statistical pass rule redesigned: 16 seeds (was 2, then 8), trimmed relMSE, two block-mean bias metrics, per-metric limits | The first version gave false failures and missed 5% errors (see Phase 1, statistical check) | 2026-11-09 |
| C7 | Two more scripts than planned: `scripts/vpt.py` (shared helpers) and `scripts/refs.py` (references + noise floor). The FLIP error map is written as `.ppm`. | Shared code in one place, and no PIL dependency | 2026-11-09 |
| C8 | Phase 2 gets an nvcc compile check of the shared headers after every step | "No host-only calls in the hot path" otherwise has no test until Phase 3 | 2026-11-12 |
| C9 | Noise floor re-measured with the new RNG after 2.8 (references unchanged) | The GPU uses the same PCG sampler, so it should be compared against a floor from that sampler | 2026-11-12 |
| C10 | Phase 2 is committed one step at a time (Phase 1 was one commit) | Clean checkpoints that are easy to bisect if the GPU port finds a problem | 2026-11-12 |
| C11 | Every speed comparison is a same-session A/B: the previous build is benchmarked right before or after the new one. The final CPU column of the timing table gets re-measured in the same session as the GPU numbers. | The identical Phase 1 code ran 6–10% faster on 2026-11-12 than on 2026-11-09, more than most single changes will gain | 2026-11-12 |
| C13 | The CPU-only direct integrator uses the new `Rng` too (the plan said it could keep mt19937) | It shares `shadeDirect` with the path tracer's NEE-on mode, and keeping mt19937 would have meant two copies. Its output is statistically unchanged | 2026-11-12 |
| C12 | A/B runs are **interleaved** per scene (`bench.py --base`), not one build after the other. Amends C11. | Running back to back favours the first build by up to 3% as the machine heats up (2.2) | 2026-11-12 |

---

## Phase 0 — repository setup (2026-11-08)

- **Bug (repo):** the ignore file was named `.gitignore.txt`, so git applied none of its rules (`out/`, `.vs/`, `*.exe`, `*.pdb`). Renamed it to `.gitignore` (`6e4f93a`) and cherry-picked that onto `main` (`eda041c`). It's the only change to `main`.
- **Decision:** `cuda-port` holds all port work. `main` stays the renderer from the published writeup.

## Phase 0.5 — plan and toolchain (2026-11-08 to 2026-11-09, `b0bb232`)

**Plan.** `PLAN.md` was written from a full read of the source and scenes. Decisions D1–D7 were settled on 2026-11-08: one executable with `--gpu`, refactors before the RNG, references in a git-ignored `refs/`, one emissive material per quad light, a bounce cap of 256, the GPU covering all of `pathTrace`, and the plan committed.

**Findings from the code read** (each one shapes a Phase 2 step):
- **Finding:** each pixel uses one `std::mt19937`, drawing numbers in sequence across all samples and bounces, so sample *s* depends on every earlier sample. A stateless sampler is needed (2.8).
- **Finding:** `Triangle`, `Sphere` and `Hit` each carry a full `Material` (~56 B), and a sphere also stores two `mat4`s (128 B). That points to material indices (2.3) and is a likely register-pressure cause later (Phase 4).
- **Finding:** quad lights aren't in the BVH. On a light hit, `traceScene` builds a `Material` on the fly, which led to D4.
- **Finding:** `pdfNEE` is O(lights) and runs lights + 1 times per vertex, so O(lights²). That matters for `mis.test` (5 lights). A Phase 4 candidate.
- **Finding:** the 8-bit conversion truncates (`(unsigned char)(c * 255)`) rather than rounding. Any new output pass has to copy that to keep PNGs byte-identical.
- **Finding:** BVH is a median split with one prim per leaf and 36-byte nodes, reached through a `Prim` indirection. All Phase 2/4 material.

**Environment.**
- **Environment:** no CUDA Toolkit or Nsight was installed. CUDA Toolkit 13.5 (nvcc V13.5.37) was installed with Nsight Compute 2026.4, Nsight Systems 2026.5.2 and compute-sanitizer. The installer offered no driver, so the driver stays at 610.88 (CUDA 13.3), and 13.5 builds run on it.
- **Environment:** VS 2026 (MSVC 19.51) is accepted as nvcc's host compiler. A forum report of 13.4.1 rejecting VS 2026 didn't apply to 13.5.
- **Environment:** Windows Smart App Control (enforce mode) blocked every unsigned binary, including all our builds and flip-evaluator's native module. It was turned off on the dev machine (Defender and SmartScreen stay on).
- **Environment:** the existing VS build folder had a CMake cache from the project's previous location, so builds go to `out/build/cli-release` (see open items).
- **Finding:** GLM 1.0.0 compiles for the device but needs `--expt-relaxed-constexpr`. Without it there are hundreds of warning #20013 from GLM calling `std::multiplies` in device code. Recorded for Phase 3's CMake setup.
- **Finding:** `nsys` isn't on PATH (use its `target-windows-x64` folder). `ncu` needs a counter permission (see open items).

**Results.**
- **Result:** the unchanged CPU build renders `scenes/cornellScatter.test` byte-identical to the committed `scenes/cornellScatter.png`, so the starting point exactly matches the published output.
- **Result:** a CUDA smoke test (GLM vector maths on the device, 1M elements) matches the CPU on every element, compute-sanitizer reports 0 errors, and Nsight Systems profiles it.

## Phase 1 — CPU baseline and harness (2026-11-09, `ae82772`)

**Renderer changes (1.1–1.3).** `--spp`, `--out`, `--seed` and `--quiet` were added. Parse, BVH build and render are timed separately and printed on one line scripts can read. The BVH build moved out of `render()`, a linear float buffer is written as `.pfm` next to the PNG, and B1 is fixed.
- **Result:** all 8 benchmark and correctness scenes at 8 spp give PNGs byte-identical to the previous build. Rebuilding the PNG from the PFM matches exactly for gamma-1 scenes. Gamma-2.2 scenes differ by 1 level in 5–13 pixels, from numpy `pow` vs MSVC `powf` rounding.
- **Finding / design note:** seed *s* uses per-pixel seeds `s·w·h + pixel`. The obvious `pixel + s` would have given seed 1 the same random streams as seed 0, shifted by one pixel, so the "independent" runs used for the noise floor would have been strongly correlated.

**Harness (1.5–1.8).** `scripts/compare.py`, `bench.py`, `check.py`, `refs.py` and the shared `vpt.py` were written, needing only numpy and flip-evaluator. The exact check (`--save-baseline` / `--exact`, about 5 s) passes on the unchanged build and fails as it should when given `--seed 1`.

**References (1.6).** 8 scenes at 4096 spp with seed 1000, 27 minutes in total. The machine wasn't idle, so the times in `refs/README.txt` are rough.
- **Finding:** `hero`'s reference still shows fireflies at 4096 spp, as the original writeup describes (in-fog paths connecting to the small, bright light). This limits how tight `hero`'s check can be.

**Statistical check: three attempts (C6).**
- **Harness, v1** (8 seeds, plain relMSE + FLIP, limit mean + 3σ): relMSE varied up to 2400× between seeds on firefly scenes (`cornellBRDF` 0.65 → 1572), so its limit was meaningless.
- **Harness, v2** (worst 0.1% of pixels dropped from relMSE): relMSE became stable (0.6–3.5% seed-to-seed). But the unchanged CPU build with a fresh seed failed 2 of 8 scenes. Over 10 seeds, FLIP on `ggx` spread about 2.5× wider than the first 8 suggested, so 8 seeds were too few. And with injected errors, even a 5% brightness error passed on the noisy `cornellBRDF` and `cornellCosine`, because per-pixel noise hides a uniform bias.
- **Harness, v3 (final):** 16 seeds, and two metrics on 16×16-pixel block means (`bias`, signed, and `blockrel`). Each has a z-score limit (4σ, with FLIP at 5σ because of its heavier tails). One fresh-seed false failure at FLIP +4.3σ led to FLIP's wider limit.
- **Result:** validated with the unchanged CPU build: **0 of 64** fresh-seed renders fail. Injected errors of 1–5% over the whole image, and 5–10% in one quarter, are caught on 7 of 8 scenes. On the two noisy Cornell scenes only the block metrics catch them. `hero` catches only ×0.98 and ×1.05 overall (open items).

**CPU baseline (1.9).** 128 spp, 20 threads, median of 5 after 1 warm-up, render time only:

| Scene | Render | Samples per second |
|---|---|---|
| cornellScatter | 4060 ms | 7.3 M |
| hero | 10381 ms | 7.9 M |
| dragon | 12851 ms (+192 ms parse, 79 ms BVH) | 3.1 M |
| mis | 5874 ms | 8.6 M |

- **Finding:** dragon is about 2.5× slower per sample than the other scenes, as expected from its 100k-triangle BVH. It's the scene where Phase 4's BVH work should show most.
- **Harness:** these rows are stamped `b0bb232+dirty` because Phase 1 wasn't committed when they ran. They match the committed `ae82772` code.

## Phase 2 — make the CPU code GPU-ready (from 2026-11-12)

Exact baseline for steps 2.1–2.7: `out/check/phase1` (git-ignored), saved from the Phase 1 build (`c203b47`, same code as `ae82772`).

### 2.1 — `HD` macro and `inline` (2026-11-12)

- New `Common.h` with the `HD` macro. All 31 functions `pathTrace` reaches are `HD inline`, plus `Ray`'s constructor and `AABB::hit`. CPU-only free functions (`arvoPhi`, `shadeAnalytic`, `shade`, `trace`, `render`, `getSphereBox`, `getTriBox`, `parseScene`, `writePFM`, `pfmPath`) are `inline`. **B3 fixed.**
- **Harness:** device compile check (C8) added as `tests/devcheck.cu` + `scripts/devcheck.py`. The kernel calls `pathTrace`, so everything it reaches is compiled for the device. The script reports nvcc errors and lists by name each host-only function still called from device code. `--strict` also fails on those, which is the Phase 2 end state.
- **Finding:** the first device compile failed with 20 errors. `EPS` and `INF` were plain `const float` globals, which nvcc won't read from device code. They're now `constexpr`, with the same values.
- **Finding:** what's left is 14 distinct host-only functions called from device code, all expected:
  - `std::mt19937` + `uniform_real_distribution` in `pathTrace`, `sampleHG`, `shadeDirect`, `shadeMIS` and `shadeMISVolume` → removed by 2.8
  - `std::vector` access to quad lights, BVH nodes, prims, spheres and triangles in `traceBVH`, `traceScene`, `pdfNEE`, `shadeDirect`, `shadeMIS` and `shadeMISVolume` → removed by 2.4/2.5
- **Finding:** `std::max` and `std::numeric_limits` don't show up, because they're `constexpr` and `--expt-relaxed-constexpr` allows them on the device. Step 2.2 still replaces them (style rule: no `std::` in the hot path), but it isn't needed to compile.
- **Result:** exact check 8/8 byte-identical. Device compile: no errors, 14 host calls left. Committed as `7beb909`.
- **Harness:** the first 2.1 benchmark was disturbed partway through by something outside the renderer: dragon's runs ranged 11.4–54.6 s and mis's 5.4–15.7 s, while the machine was idle again afterwards. The run was repeated and its 4 rows removed from `bench/results.csv`.
- **Finding:** the clean re-run was 3–12% faster than the Phase 1 baseline from 2026-11-09. To find out why, the Phase 1 code (`ae82772`, exported with `git archive` and built separately) was timed in the same session:

  | Scene | Phase 1 build | 2.1 build | Change |
  |---|---|---|---|
  | cornellScatter | 4044 ms | 3935 ms | −2.7% |
  | hero | 9557 ms | 9362 ms | −2.0% |
  | dragon | 11509 ms | 11353 ms | −1.3% |
  | mis | 5545 ms | 5416 ms | −2.3% |

  **Result (withdrawn, see 2.2):** this A/B ran the builds one after the other, with Phase 1 second, and 2.2 showed that the second build in such a run comes out slower. So the "2.1 is about 2% faster" reading is not trustworthy. The interleaved re-run (see Phase 2 benchmarks) found no clear effect. Most of the gap to the 2026-11-09 baseline is the machine: the identical Phase 1 code runs 6–10% faster today. Timings from different sessions can't be compared (C11).
- **Harness:** `bench.py` gained `--commit`, to stamp rows from a binary that isn't HEAD (A/B runs). The `+dirty` flag now looks only at renderer source (`*.h`, `*.cpp`, `*.cu`, `CMakeLists.txt`), because appending to `bench/results.csv` made every later run "dirty". The 2.1 rows were corrected from `7beb909+dirty` to `7beb909`.

### 2.2 — no STL in the hot path (2026-11-12)

- 10 `std::max` calls in path-tracer functions became `maxf` (new in `Common.h`, `a < b ? b : a`, the same as `std::max` including NaN handling, unlike `fmaxf`). `INF` (`numeric_limits<float>::max()`) was replaced by `FLT_MAX` directly, and the `AABB` defaults use `FLT_MAX` too. The 6 `std::max` calls in the CPU-only Whitted `shade()` stay.
- **Result:** exact check 8/8 byte-identical. Device compile clean, still 14 host calls (`std::max` was never among them).
- **Finding: benchmark order effect.** The first A/B ran the builds one after the other (2.1, then 2.2) and showed 2.2 3.0% slower on cornellScatter, with no overlap between the two min/max ranges, even though nothing in the code should change speed. Re-run **interleaved** per scene (base, test, test, base, …):

  | Scene | 2.1 build | 2.2 build | Change |
  |---|---|---|---|
  | cornellScatter | 4057 ms | 4063 ms | +0.1% |
  | hero | 9567 ms | 9656 ms | +0.9% |
  | dragon | 11593 ms | 11628 ms | +0.3% |
  | mis | 5475 ms | 5492 ms | +0.3% |

  **Result:** 2.2 is neutral, within noise. The 3% was entirely from run order: the machine slows down as it heats over minutes of full load, so whichever build runs second loses. The 2.1 A/B had the same flaw (above).
- **Harness:** `bench.py --base EXE --base-commit C` times a base build interleaved with the build under test, alternating the order of each pair, and prints the change. The 16 earlier rows from back-to-back A/B runs keep their numbers but are labelled `[sequential A/B, order-biased]` in `bench/results.csv`. Builds for A/B are kept in `out/check/bin/`, named by commit (`8ff144a.exe` = 2.1, `a0cef85.exe` = 2.2), or by step until committed.
- Committed as `a0cef85`.

### 2.3 — material indices (2026-11-12)

- `Scene` holds a flat `materials` array. `Triangle`, `Sphere` and `Hit` carry an `int matId` in place of a full `Material`. The parser deduplicates (`materialId`, exact comparison via `Material::operator==`, newest first). Each quad light gets its own emissive material at the end of parsing (D4), and `traceScene` sets that light's `matId` in place of building `Material{}` on every light hit. Every function that read `h.mat` now takes `const Material& hm = sc.materials[h.matId]`. The renderer's scene line now also prints the material count.
- **Result:** exact check 8/8 byte-identical. Device compile clean. Host calls are up from 14 to 15, the extra one being the new `std::vector<Material>` access, which 2.4 removes with the other vectors.
- **Result:** struct sizes (MSVC x64):

  | Struct | Before | After |
  |---|---|---|
  | `Hit` | 92 B | 36 B |
  | `Triangle` | 108 B | 52 B (dragon: 10.8 MB → 5.2 MB of triangles) |
  | `Sphere` | 204 B | 148 B (two `mat4`s left, see Phase 4) |
  | `QuadLight` | 48 B | 52 B |

  Unique materials per scene: cornellScatter / cornellBRDF / cornellCosine / cornellFog 5, hero 7, dragon 4, mis 11, ggx 9.
- **Finding:** in `trace()` (the Whitted integrator), `h` is a local declared after a depth check, so the material reference has to come after the miss check. The first scripted edit put it at the top of the function, and a read of the result caught it before building.
- Benchmark pending (machine busy). Committed as `0722b41`, and the build is kept as `out/check/bin/0722b41.exe` to A/B against `a0cef85.exe` when the machine is idle.

### 2.4 — `SceneView` (2026-11-12)

- New `SceneView.h`: a plain struct of raw pointers + counts (spheres, tris, materials, BVH nodes, prims, quad / directional / point lights), plus camera, medium and integrator settings. Field names match `Scene`, so the shading code reads the same. `makeView(scene, bvh)` builds it, pointing into the vectors, which must outlive it.
- Every function in `Raytracer.h` except `render()` takes `const SceneView& sc` in place of `const BVH& bvh, const Scene& sc`. `render()` keeps its signature and builds the view once. Range-based loops over lights became index loops (`qi` / `li`, because `shadeDirect` already uses `k` for its light samples).
- **Decision:** directional and point lights (+ `atten`) are in the view too, though only the CPU-only Whitted integrator reads them, so all of `Raytracer.h` takes one scene type. The GPU upload will leave them null.
- **Result:** exact check 8/8 byte-identical. Device compile clean, and **host calls fell from 15 to 2**: only `std::mt19937` / `uniform_real_distribution` remain, and 2.8 removes them. `tests/devcheck.cu` now passes `SceneView` by value as a kernel argument, the way Phase 3 will.
- Benchmark pending (machine busy). The build is kept as `out/check/bin/2.4.exe`.
- **Harness:** when starting 2.5, the 2.4 build was renamed to `<HEAD>.exe` without first checking that 2.4 had been committed. HEAD was still 2.3, so this overwrote the kept 2.3 build. The 2.3 build was rebuilt from `0722b41` with `git archive` (exact check 8/8). Builds are now renamed only after confirming HEAD has moved.
- Committed as `c6fd115`.

### 2.5 — BVH leaves index primitives directly (2026-11-12)

- **Finding:** `Prim` was an identity mapping. `BVH::build` adds every sphere and then every triangle in input order, so `prims[i]` was always (sphere `i`) or (triangle `i − numSpheres`). Traversal paid an extra 32-byte load per leaf for nothing.
- New `BVH::reorder(scene)`, called right after `build`, rewrites `scene.spheres` and `scene.tris` into leaf order (the nodes are stored depth-first, so neighbouring leaves now hit neighbouring primitives in memory). Each leaf's `prim` then indexes them directly: values below `numSpheres` are spheres, the rest are triangles after them. `traceBVH` reads `spheres[prim]` / `tris[prim − numSpheres]`. `Prim` is now build-only, and `SceneView` no longer has `prims`. The node layout is unchanged (layout work is Phase 4).
- **Result:** exact check 8/8 byte-identical. No hit-distance ties changed which primitive wins, so the possible exception noted in the plan didn't occur. Device compile clean, still 2 host calls (RNG only). The reorder adds no measurable BVH build time (dragon ~73 ms both before and after, 3 runs each).
- Benchmark pending. The build is kept as `out/check/bin/2.5.exe`. The leaf-order memory layout is the part most likely to show up in timing, on dragon.
- Committed as `a110fc0`.

### 2.6 — ray–box NaN (B2) (2026-11-12)

- **Finding:** first measured how often B2 actually happens. A scratch build of `a110fc0` counted NaN `tenter`/`texit` in `AABB::hit` over all 8 scenes at 8 spp: **0 NaNs in about 3.5 billion box tests**. At most one ray per scene had an exactly-zero direction component, and none of those also started exactly on a box face, which NaN needs. So B2 is latent in these scenes.
- **Finding:** a scratch test of exactly that case (unit box, ray starting on the y = 0 face, moving along +x) shows why it still matters. The old code gets `t0.y = 0 × ∞ = NaN`, and the answer then depends on argument order: `glm::min(t0, t1)` gives a **hit**, `glm::min(t1, t0)` a **miss**. The GPU's `fminf`/`fmaxf` skip NaN, so CPU and GPU could disagree on such rays.
- **Fix:** new `safeInv` in `BVH.h`: `1/x`, with `|x| < 1e-20` nudged to `±1e-20`. `traceBVH` builds the inverse direction with it, so it's always finite and `AABB::hit` can't produce NaN. In the test above, both argument orders then agree (hit). **B2 fixed.**
- **Result:** exact check 8/8 byte-identical (expected, given 0 NaNs before). Device compile clean, still 2 host calls (RNG only).
- Benchmark pending. The build is kept as `out/check/bin/2.6.exe`.
- Committed as `7fcc5c0`.

### 2.7 — `renderPixel` and output pass (2026-11-12)

- New `renderPixel(sc, x, y, s, rng)` (`HD`): the radiance of one sample, with sample 0 through the pixel centre. Until 2.8 it still takes the pixel's mt19937, because sample *s* depends on the numbers earlier samples drew. 2.8 drops that parameter.
- `render()` now only adds samples into a per-pixel `sum` buffer (zeroed by the caller), in the same order as before (s = 0..spp−1). It returns the samples per pixel: spp for the path tracer, 1 for the other integrators, which store their single value. The camera ray at the pixel centre is now only computed by the integrators that use it.
- New `outputPass()` in `Image.h`: `sum / n` → linear average (written as the PFM) → clamp → gamma → truncate to 8 bits, exactly the old per-pixel code. It divides (`vec3 / float`), as before, rather than multiplying by `1/n`, which could change the last bit. `render_ms` no longer includes this pass, which is a trivial loop.
- **Result:** exact check 8/8 byte-identical. Device compile clean, still 2 host calls (RNG). `tests/devcheck.cu` now calls `renderPixel`, the function Phase 3's kernel will call.
- **Result:** the 8 check scenes all use the path tracer, but 2.7 also touched the Whitted, analytic-direct and direct branches. Scratch copies of `cornellBRDF` switched to `raytracer`, `analyticdirect` and `direct` (plus a point light, a directional light and 4 stratified light samples) give byte-identical PNG and PFM on `7fcc5c0` and 2.7.
- Benchmark pending. The build is kept as `out/check/bin/2.7.exe`.
- Committed as `b683983`.

### 2.8 — stateless RNG (2026-11-12)

- New `Rng.h`: `pcgHash` (one PCG step plus its RXS-M-XS output permutation, after Jarzynski & Olano 2020) and `Rng`, whose numbers depend only on (seed, pixel, sample, dimension). `key = hash(sample + hash(pixel + hash(seed)))`, and `next()` returns the top 24 bits of `hash(key ^ dim·0x9e3779b9)` scaled to [0, 1). Spreading `dim` by the golden ratio before the XOR means two samples whose keys happen to be close can't produce shifted copies of the same sequence, as `key + dim` could.
- `std::mt19937`, `uniform_real_distribution` and `<random>` are gone from `Raytracer.h`. `renderPixel(sc, x, y, s)` builds its own `Rng` from `sc.seed`, the pixel index and `s`, so sample *s* no longer depends on what earlier samples drew. That's what lets Phase 3 run one launch per sample. Sample 0 still goes through the pixel centre. `SceneView` carries `seed`.
- **Plan change (C13):** the CPU-only direct integrator also uses `Rng`. The plan let it keep mt19937, but it shares `shadeDirect` with the path tracer's NEE-on mode, and keeping both would have meant two copies of `shadeDirect`. Its images change statistically only: on a scratch direct-lighting scene with 64 light samples, PCG vs mt19937 gives relMSE 0.000365 and bias +0.011%, the same as mt19937 against itself with another seed (0.000366, −0.001%).
- **Result:** `devcheck.py --strict` passes with **no diagnostics at all**. The whole path tracer compiles for the GPU with no host calls (14 at 2.1, 15 at 2.3, 2 at 2.4).
- **Result:** statistical check against the references and the mt19937 noise floor: **8/8 pass, every z-score within ±2**.
- **Result (C9):** noise floor re-measured with PCG (16 seeds × 8 scenes), with the mt19937 floor kept as `refs/floor_mt19937.json`. A Welch t-test on each scene and metric (16 + 16 samples) finds no difference: largest |t| is 2.72 over 32 comparisons, under the ~3.0 that chance alone allows (Bonferroni, 5%). It's on `mis`'s bias, −0.0016% vs +0.0026%. `refs/floor.json` is now the PCG floor (also kept as `refs/floor_pcg.json`).
- **Result:** check validation repeated against the PCG floor: **0 of 64** fresh-seed renders fail. Detection matches the mt19937 floor except `cornellCosine`: its bias spread came out slightly wider (0.25% vs 0.21%), so ×1.01 overall and ×1.05 in one quarter now pass, while ×1.02, ×0.98, ×1.05 overall and ×1.10 in a quarter are still caught. `hero` is as before (×0.98 and ×1.05 overall only).
- New exact baseline `out/check/phase2-pcg` for 2.9 onwards, since every image changes at this step. It reproduces itself 8/8.
- Benchmark pending. The build is kept as `out/check/bin/2.8.exe`. PCG vs mt19937 on the CPU is a speed result worth having.
- Committed as `15df063`.

### 2.9 — bounce cap (B4) (2026-11-12)

- **Finding:** path lengths measured first, with a scratch build of `15df063` recording each path's bounce count over all 8 scenes at 64 spp: **172 million paths, longest 100 bounces** (`ggx`), 6 paths over 64, none near 256. The tail per scene:

  | Scene | maxdepth | Longest | Paths > 16 | > 32 | > 64 |
  |---|---|---|---|---|---|
  | cornellScatter | −1 | 45 | 24,659 | 41 | 0 |
  | hero | −1 | 21 | 31 | 0 | 0 |
  | dragon | 5 | 5 | 0 | 0 | 0 |
  | mis | 1 | 1 | 0 | 0 | 0 |
  | cornellBRDF | −1 | 41 | 7,804 | 7 | 0 |
  | cornellCosine | −1 | 27 | 1,406 | 0 | 0 |
  | cornellFog | −1 | 27 | 258 | 0 | 0 |
  | ggx | −1 | 100 | 7,074 | 299 | 6 |

- `pathTrace` stops after `MAX_BOUNCES = 256` bounces, also when `maxdepth` is −1 (unbounded) or above 256. **B4 fixed.**
- **Result:** exact check 8/8 byte-identical against `out/check/phase2-pcg` (as the measurement predicted: no path reached the cap). `devcheck --strict` still clean.
- **Finding (Phase 4 lead):** with one thread per pixel, a warp of 32 threads runs until its longest path ends. Typical paths in the unbounded scenes are a handful of bounces, but `ggx` has paths up to 100 and `cornellScatter` up to 45, so expect idle threads in those scenes. That's the "threads in the same group taking different branches" item, now with numbers to check against Nsight Compute.
- Benchmark pending. The build is kept as `out/check/bin/2.9.exe`.

### Phase 2 summary (code complete 2026-11-12)

- The path tracer compiles for the GPU: `devcheck --strict` is clean. Every function `renderPixel` reaches is `HD`, reads the scene through `SceneView` (raw pointers + counts), and draws random numbers from the stateless `Rng`.
- Steps 2.1–2.7 and 2.9 were byte-identical against their baselines. 2.8 (RNG) changed every image and passed the statistical check, and its noise floor matches mt19937's.
- Bugs fixed: B2 (ray–box NaN), B3 (header functions not `inline`), B4 (no bounce cap). Plan changes: C8–C13.
- Measurement lessons: same-session A/B (C11) and interleaved A/B (C12).
- **Still owed before Phase 3:** the CPU benchmarks for 2.3–2.9 (interleaved, idle machine) and a post-Phase-2 CPU timing run for the timing table. The renderer has no open correctness issues.

### Phase 2 benchmarks (2026-11-12, 22:41–23:35)

All builds were first re-verified against their exact baselines (8/8 each). Then 9 interleaved A/B runs (`bench.py --base`, 1 warm-up + 5 runs each per build, 128 spp), one after another on an idle machine. Change in median render time against the previous step (negative = faster). \* = the two builds' min/max ranges don't overlap:

| Step | cornellScatter | hero | dragon | mis |
|---|---|---|---|---|
| 2.1 HD + inline (recheck, vs Phase 1) | −0.8% | −2.6% | −0.4% | −1.3% |
| 2.3 material indices | −0.1% | −2.3%\* | −0.6% | −0.1% |
| 2.4 SceneView | −1.1% | −0.2% | −1.1% | −1.8%\* |
| 2.5 BVH leaves index prims directly | +0.0% | −1.0%\* | **−3.9%\*** | +0.4% |
| 2.6 ray–box NaN fix | −0.1% | −0.7% | −0.9%\* | −1.4% |
| 2.7 renderPixel + output pass | −1.1% | +0.8% | +0.7% | −3.2% |
| 2.8 PCG RNG | **−5.2%\*** | −1.9% | −0.7% | −1.3% |
| 2.9 bounce cap | −0.3% | +0.1% | −1.0% | −0.4% |
| **Phase 2 total (Phase 1 vs `183b237`)** | **−7.3%\*** | **−5.9%\*** | **−7.4%\*** | **−6.3%\*** |

(2.2 was measured on its own, interleaved: +0.1% to +0.9%, no change.)

- **Result:** Phase 2 made the CPU renderer **6–7% faster on every scene**, clearly outside the noise, even though no step aimed at CPU speed.
- **Result:** the steps that stand out have a clear mechanism:
  - **2.8 PCG on cornellScatter, −5.2%:** that scene draws the most random numbers per path (free-flight sampling and phase sampling in the medium), and a few integer multiplies are cheaper than mt19937 plus `uniform_real_distribution`.
  - **2.5 on dragon, −3.9%:** dropping the per-leaf `Prim` load and storing triangles in leaf order matters most in the 100k-triangle scene.
  - **2.3 on hero, −2.3%:** the smaller `Hit` (92 → 36 B) is copied on every closest-hit update.
  - The other steps are within about ±1%, as expected for pure refactors.
- **Result:** the 2.1 recheck, now interleaved, gives −0.4% to −2.6% with all ranges overlapping. So 2.1 has no clear effect, which replaces the withdrawn "2% faster" claim.
- **Finding:** the step-by-step changes roughly add up to the total (e.g. cornellScatter −8.7% summed vs −7.3% measured directly), which suggests the per-step numbers are consistent. Only the direct Phase 1 vs Phase 2 comparison is used as the headline.
- End-of-Phase-2 CPU timings (same session as the total): cornellScatter 3734 ms (7.90 Ms/s), hero 8840 ms (9.27 Ms/s), dragon 10323 ms (3.81 Ms/s), mis 5056 ms (9.96 Ms/s).
- Committed as `2a9d2d6`.

## Phase 3 — simple CUDA kernel (from 2026-11-13)

### 3.1 — kernel, build and correctness (2026-11-13)

- **CMake:** option `VPT_CUDA` (default on). If `check_language(CUDA)` finds nvcc, it enables CUDA, adds `Kernels.cu`, sets `CUDA_ARCHITECTURES 86`, C++17 and `--expt-relaxed-constexpr` (CUDA sources only), and defines `VPT_HAS_CUDA`. Otherwise the build is the CPU renderer, unchanged. `cmake_minimum_required` went from 3.15 to 3.18 (needed for `CUDA_ARCHITECTURES`). Detected: nvcc 13.5.37 with host compiler MSVC 19.51.
- **New files:**
  - `CudaCheck.h`: `CUDA_CHECK(call)` prints error name, string, the call, file and line, then exits.
  - `GpuRender.h`: host-facing `gpuInit()` / `renderGPU()` with no CUDA types, so `main.cpp` stays plain C++.
  - `Kernels.cu`: `renderKernel`, one thread per pixel, 16×16 blocks, `sum[p] += renderPixel(sc, x, y, s)`. `renderGPU` uploads each `SceneView` array with `cudaMalloc` + `cudaMemcpy` (the directional and point lights, used only by Whitted, are left out), zeroes the sum buffer, launches once per sample, times the launches with CUDA events, downloads, and frees everything in one place. Every runtime call and every launch (`cudaGetLastError`) goes through `CUDA_CHECK`.
- **`main.cpp`:** `--gpu` (path tracer only; a CPU-only build refuses it with a message). Context creation (`gpuInit`) happens outside the timings. The GPU run reuses the same `outputPass` and PNG/PFM writers. The timing line keeps `render_ms` (CUDA events: all sample launches) and adds `upload_ms`, `download_ms` and `total_ms` (wall clock for upload + render + download) in place of `threads`.
- **Result:** both configurations build. The CUDA build compiled and linked first time. The CPU-only build (`-DVPT_CUDA=OFF`) and the CUDA build both pass the exact check 8/8 on the CPU path (`out/check/phase2-pcg`).
- **Result:** `compute-sanitizer --tool memcheck` at 2 spp on cornellScatter, dragon and ggx: **0 errors** each.
- **Result:** statistical check of the **GPU** against the CPU references and PCG noise floor: **8/8 pass**, every z-score within ±2.1. This is Phase 3's correctness criterion.
- **Result:** with the same seed, CPU and GPU draw the same random numbers, so they can be compared pixel by pixel (64 spp):

  | Scene | Pixels within 1e-5 relative | Within 1e-3 | Image mean CPU vs GPU |
  |---|---|---|---|
  | cornellScatter | 90.4% | 99.9% | +0.001% |
  | hero | 99.0% | 99.9% | +0.000% |
  | dragon | 98.8% | 99.8% | +0.001% |
  | mis | 98.7% | 100.0% | +0.002% |
  | cornellBRDF | 92.1% | 99.8% | +0.001% |
  | cornellCosine | 90.3% | 99.9% | +0.000% |
  | cornellFog | 88.8% | 99.9% | +0.000% |
  | ggx | 77.4% | 94.4% | +0.010% |

  The remaining differences come from float rounding: nvcc contracts multiply-adds into FMAs, and its `sinf`/`powf`/`logf` differ in the last bit, which now and then flips a Russian-roulette or free-flight decision and sends that one path elsewhere. `ggx` matches least because sharp GGX lobes amplify tiny direction differences.
- **Result (indicative only, machine not confirmed idle):** first GPU run of cornellScatter at 128 spp: render 720 ms (41 Ms/s), upload 2.0 ms, download 0.3 ms, against 3734 ms on the CPU at the end of Phase 2. That's about 5× from the naive kernel. Proper benchmarks come in 3.2.
- **Harness:** `check.py`/`bench.py` need `--extra=--gpu` (with `=`). Written as `--extra "--gpu"`, argparse takes `--gpu` for an option of its own and errors.
- Build kept as `out/check/bin/3.1.exe`, which has both backends.
