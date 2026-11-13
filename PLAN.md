# VPT CUDA port — detailed plan

Working plan for porting the VPT path tracer to CUDA: what the code looks like today, the design decisions, each phase's tasks, how each step is checked, and the risks. What actually happened (bugs found, plan changes, results) is logged in `PROGRESS.md`.

Status: **Phase 1 done (2026-11-09): harness, references, validated statistical check, CPU timings (section 8). Phase 2 in progress: 2.1–2.3 done 2026-11-12 (2.3 benchmark pending). Next: 2.4.** Phases run one at a time, with a review before each.

---

## 1. Starting point

### Code today (`cuda-port` at `6e4f93a`, same renderer as `main`)

| File | Lines | What matters for the port |
|---|---|---|
| `main.cpp` | 39 | Parses the scene, calls `render()`, writes the PNG. No CLI options besides the scene path. |
| `Scene.h` | 108 | `Scene` is all `std::vector`/`std::string`. `Triangle`, `Sphere` and `Hit` each carry a full `Material` (4 × vec3 + 2 floats + enum ≈ 56 B). A sphere also stores two `mat4`s (128 B). |
| `BVH.h` | 138 | Median split on the longest axis, **one prim per leaf**, nodes reach prims through `Prim {kind, idx, box}`. Node = AABB + 3 ints (36 B). `AABB::hit` can make NaN (0 × inf). |
| `Raytracer.h` | 942 | Every integrator. The path tracer reaches: `traceScene`, `traceBVH`, `hitSphere`, `hitTriangle`, `hitQuad`, `shadeDirect` (NEE on), `shadeMIS`, `shadeMISVolume`, `sampleHG`, `hgPhase`, the Phong/GGX eval/sample/pdf functions, `pdfNEE`, `transmittance`, `onbBasis`, `makeRay`. |
| `Parser.h` | 222 | Host-only, stays as it is. |

Facts that affect the plan:
- **RNG:** one `std::mt19937` per pixel, seeded with `j * w + i`, used one number after another across all samples and bounces. How many numbers a path uses varies (RR, scatter events, number of lights), so sample `s` of a pixel depends on everything sampled before it. A stateless sampler fixes that, but it changes every image, so it has to be checked statistically (see 2.8).
- **Not marked `inline`:** functions are defined in headers without `inline`. That only works now because just `main.cpp` includes them. Once a `.cu` file includes the same headers, linking will fail with duplicate symbols. The `HD` macro change must add `inline` too.
- **The hot path uses STL:** `std::max` (BRDFs and pdfs), `std::numeric_limits` (`INF`, `AABB` defaults), loops over `sc.quadLights` (a vector), `std::uniform_real_distribution`.
- **Quad lights are outside the BVH.** `traceScene` tests every quad after the BVH, and on a hit it builds a `Material` on the fly with `emission = intensity`. With material indices, that needs a home (decision D4).
- **`pdfNEE` is O(lights)** and runs once per light plus once for the BRDF sample at every vertex, so O(lights²) per vertex. That matters for `mis.test` (5 lights). It's a Phase 4 candidate, not a Phase 2 change.
- **`render()` builds the BVH itself** and prints progress from inside the parallel loop (`done++` is a data race). For timing, parse, BVH build and render have to be split.
- **Output:** `clamp(0,1)`, then `pow(1/gamma)` when gamma ≠ 1, then `(unsigned char)(c * 255)` (truncates, doesn't round). The new output pass must reproduce this exactly so the PNGs stay byte-identical.
- **No bounce cap:** with `maxdepth -1`, only RR ends a path. That's fine on the CPU, but it gives long-tailed warps on the GPU (decision D5).

### Machine

| | |
|---|---|
| GPU | RTX 3070 Ti, 8 GB, sm_86, driver 610.88 |
| CPU | i7-12700KF, 12 cores (8P + 4E) / 20 threads |
| IDE | Visual Studio Community 2026 (18.7), CMake + Ninja come with it (not on PATH) |
| GLM | 1.0.0 in `include/glm`. It recognises nvcc (`__CUDACC__`) and marks its functions `__host__ __device__` |
| Python | 3.12.10, numpy 2.5.3 |
| CUDA | Toolkit 13.5 (nvcc V13.5.37) with MSVC 19.51 (VS 2026) as host compiler |
| Profilers | Nsight Compute 2026.4, Nsight Systems 2026.5.2 (`nsys` not on PATH: use the `target-windows-x64` folder of its install), compute-sanitizer 13.5 |

The 12700KF mixes P and E cores, so CPU timings are noisier than on a uniform CPU. Always record the thread count and use the median of several runs.

---

## 2. Decisions

Settled 2026-11-08. To change one later, edit its row and note the date.

| # | Question | Decision |
|---|---|---|
| D1 | One executable with `--gpu`, or separate CPU and GPU executables? | **One executable.** CMake option `VPT_CUDA` (on when a CUDA compiler is found) links the CUDA backend in. Without it you get the same OpenMP build as today. Parsing, output and the harness stay the same for both, and `--gpu` is the only switch. |
| D2 | Order of the Phase 2 changes | **Pure refactors first, RNG last.** Before the RNG swap, every refactor can be checked **bit for bit** (same PFM hash), because each pixel's mt19937 sequence doesn't depend on thread scheduling. That's a much stronger and cheaper check than statistics. Only the final RNG swap needs the statistical check. |
| D3 | Where do the reference images go? | **`refs/` in the repo, git-ignored** (add `refs/` and `*.pfm` to `.gitignore`). Short paths for the scripts, never committed. |
| D4 | Material for quad-light hits once `Hit` holds an index | **When the scene is built, add one emissive material per quad light** to the flat material array. Then `Hit` always carries a valid `matId` and the shading code doesn't change. |
| D5 | Hard bounce cap for `maxdepth -1`? | **Add a shared cap of 256** in `pathTrace`, CPU and GPU alike, and confirm the images don't change (with RR on, paths that long have practically zero weight). It bounds the worst case per thread and avoids the TDR risk (see risks). |
| D6 | How much of the path tracer the GPU covers | **All of `pathTrace`:** NEE off/on/MIS, hemisphere/cosine/BRDF sampling, absorption-only and scattering media, Phong and GGX. It's one function, and leaving modes out means `if (gpu && …)` checks in shared code. |
| D7 | Keep the plan in the repo? | **Yes.** The plan, `PERF_LOG.md` and the benchmark data together record how the port was done. |

---

## 3. Phase 0.5 — plan and toolchain (done)

- [x] Plan written, decisions D1–D7 settled.
- [x] **CPU baseline reproduces the published output:** an unchanged RelWithDebInfo build (MSVC 19.51, OpenMP 2.0) renders `scenes/cornellScatter.test` in 4.3 s wall clock, byte-identical to the committed `scenes/cornellScatter.png`. Progress output repeats lines (the `done++` race, fixed in 1.2).
- [x] **CUDA toolchain verified:** CUDA 13.5 accepts VS 2026's MSVC as host compiler. A smoke-test kernel using GLM on the device runs on the 3070 Ti and matches the CPU result on all 1M elements. compute-sanitizer memcheck reports 0 errors, and Nsight Systems profiles it.
- [x] **GLM needs `--expt-relaxed-constexpr` under nvcc.** Without it there are hundreds of warning #20013, from GLM calling the constexpr host function `std::multiplies` in device code.
- [x] `flip-evaluator` installed for Python 3.12.
- [ ] **Before Phase 4:** allow GPU performance counters for non-admin users (`ncu` fails with `ERR_NVGPUCTRPERM`): NVIDIA Control Panel → Desktop → Enable Developer Settings, then Developer → Manage GPU Performance Counters → "Allow access to all users". The alternative is running `ncu` as admin.

---

## 4. Phase 1 — CPU baseline and harness

Goal: measure and compare before changing anything that renders. **No change in this phase may alter the 8-bit PNG output.**

### 1.1 CLI
Hand-written argument parsing in `main.cpp`, no dependency:
- `--spp N` overrides the scene's spp
- `--out path.png` overrides the output path. The PFM goes next to it, as `path.pfm`
- `--seed N` offsets the per-pixel seed by `N·w·h` (default 0 gives today's images bit for bit). This is how the **noise floor** is measured (1.5).
- `--quiet` turns off the progress output for benchmark runs
- (Phase 3) `--gpu`

### 1.2 Timing
- Time parse, BVH build and render separately with `std::chrono::steady_clock`. Move the BVH build out of `render()` and into `main` (same work, it just runs before the render).
- At the end, print one line a script can read:
  `time parse_ms=… bvh_ms=… render_ms=… spp=… threads=… msamples_s=…`
  where `msamples_s = w*h*spp / render_s / 1e6`.
- Fix the `done++` race: make it `std::atomic<int>`, and with `--quiet` skip the printing altogether.

### 1.3 Float output
- Add an accumulation buffer `std::vector<vec3>` (linear, before clamping) next to the 8-bit buffer. `render()` writes both. (The real split into `renderPixel` + output pass comes in Phase 2. This is the minimum needed.)
- Write the PFM: header `PF\n<w> <h>\n-1.0\n` (negative = little-endian), **rows stored bottom to top**, 3 × float32 per pixel. A small `writePFM` in a new `Image.h`.

**1.1–1.3 result:** all 8 benchmark + correctness scenes at 8 spp, run through `--spp`/`--out`, give PNGs byte-identical to the unchanged build. Rebuilding the 8-bit image from the PFM with clamp/gamma/truncation matches exactly for gamma-1 scenes. Gamma-2.2 scenes differ by 1 level in 5–13 pixels, from numpy `pow` vs MSVC `powf` rounding, so the data and row order are right. `--seed 0` reproduces the original image and `--seed 1` differs. Seed `s` uses per-pixel seeds `s·w·h + pixel`, so different seeds never share a stream.

### 1.4 Benchmark and correctness sets

Benchmark set (timed):

| Scene | Res | Scene spp | Bench spp | What it stresses |
|---|---|---|---|---|
| `scenes/cornellScatter.test` | 480² | 128 | 128 | scattering medium, MIS in volume |
| `scenes/hero.test` | 800² | 4096 | 128 | medium + GGX + tiny bright light (fireflies) |
| `scenes/dragon.test` | 640×480 | 128 | 128 | 100k tris, BVH traversal, depth 5 |
| `scenes/mis.test` | 768×512 | 32 | 128 | 5 lights, direct only (depth 1), NEE cost |

**Bench spp is 128 for every scene.** The 8-spp timings put each CPU run at about 5–15 s, long enough to time reliably and still long enough for a meaningful GPU timing. A full bench pass (1 warm-up + 5 runs, 4 scenes) takes about 4 minutes.

Correctness set (not timed, low spp, covers the code paths the benchmarks miss): `cornellBRDF` (NEE on + Phong, `shadeDirect`), `cornellCosine` (cosine sampling), `cornellFog` (absorption only), `ggx` (transformed spheres, GGX + Phong).

### 1.5 Comparison script — `scripts/compare.py`
Inputs: test PFM and reference PFM (plus the scene's gamma). Outputs:
- **RMSE** and **relMSE** (`mean((t-r)² / (r² + 1e-2))`, so fireflies in dark areas don't dominate) on linear values
- **mean LDR-FLIP**, after applying the renderer's own clamp + gamma to both images in the script (no 8-bit rounding)
- an optional FLIP error map (`--flipmap`), written as `.ppm` so the scripts need nothing beyond numpy and flip-evaluator

Shared code (scene sets, spp constants, renderer lookup, PFM reading, metrics) lives in `scripts/vpt.py`.

Plus two **bias metrics** on 16×16-pixel block means. Averaging 256 pixels shrinks the noise about 16×, while a bias survives:
- **bias**: signed mean relative error of the blocks (overall brightness bias)
- **blockrel**: mean squared relative error of the blocks (bias in one region)

Both drop the worst 1% of blocks, and relMSE drops the worst 0.1% of pixels, so fireflies don't dominate.

**Pass rule (statistical):** for each scene, render with **16 seeds** (0–15) at the stat spp (64) and measure every metric against the reference. That spread is the **noise floor**, stored in `refs/floor.json` by `scripts/refs.py --floor`. A candidate (Phase 2 RNG, GPU) passes a scene if it has no NaN/inf pixels and every metric is within the floor: relMSE and blockrel at most mean + 4σ, |bias − mean| at most 4σ, FLIP at most mean + 5σ (its seed-to-seed distribution has heavier tails). That replaces guessing absolute thresholds.

**How the rule was arrived at (2026-11-09):**
- *First version* (8 seeds, untrimmed relMSE + FLIP, mean + 3σ): relMSE varied up to 2400× between seeds on firefly scenes (`cornellBRDF` 0.65 → 1572), which made it useless. The unchanged CPU build failed 2 of 8 scenes with a fresh seed, because 8 seeds underestimated FLIP's spread about 2.5× (`ggx`). And on noisy scenes, even a 5% brightness error passed.
- *Final version*, validated with the unchanged CPU build: **0 of 64** fresh-seed renders (seeds 100–107 × 8 scenes) fail. With deliberate errors injected into a seed-100 render:

  | Error injected | Caught on |
  |---|---|
  | whole image ×1.01, ×1.02, ×0.98, ×1.05 | 7 of 8 scenes (all except `hero`); `hero` catches ×0.98 and ×1.05 |
  | top-left quarter ×1.05, ×1.10 | 7 of 8 scenes (all except `hero`) |

  On the noisy `cornellBRDF` and `cornellCosine`, only the block metrics catch these errors. FLIP and relMSE alone would miss them all.
- **Known limit:** `hero` keeps fireflies even in its 4096-spp reference, so its check only catches errors of about 5% or more. The same medium and volume-MIS code runs in `cornellScatter`, which catches 1%.

### 1.6 References
- `scripts/refs.py` renders each benchmark and correctness scene at **4096 spp** (32× bench spp, 64× stat spp) into `refs/<scene>.pfm`, and logs spp, seed, time and commit to `refs/README.txt` (git-ignored with the rest of `refs/`).
- Rendered 2026-11-09 in 27 min total (cornellScatter 152 s, hero 358 s, dragon 447 s, mis 208 s, the four correctness scenes 91–145 s each). The machine wasn't idle, so these times are only a rough guide.
- **References use seed 1000.** With seed 0, a 64-spp test would reuse the reference's first 64 samples per pixel (same mt19937 stream) and look better than it is.
- The references come from the **Phase 1** code (mt19937). They're the fixed point for everything after.

### 1.7 Benchmark script — `scripts/bench.py`
Runs the executable over the benchmark set, 1 warm-up + 5 timed runs each, takes the median, and appends to `bench/results.csv` (`date, commit, backend, label, scene, w, h, spp, threads, runs, parse_ms, bvh_ms, render_ms, render_min_ms, render_max_ms, msamples_s`). `--label` notes the change being measured, and `--extra "--gpu"` will select the GPU in Phase 3. The commit hash comes from `git rev-parse --short HEAD` (read-only). `bench/` stays in git, since the CSV is the raw data behind the writeup.

### 1.8 Check script — `scripts/check.py`
Runs the benchmark + correctness sets and does one of two checks:
- `--save-baseline DIR` then `--exact DIR`: renders at 8 spp and byte-compares each PNG and PFM with the saved ones (the [exact] check in Phase 2). About 5 s for all 8 scenes.
- `--stat`: renders at 64 spp with seed 0 and applies the noise-floor rule against `refs/` (the [stat] check in Phases 2–4)

Prints one pass/fail line per scene and exits non-zero on failure. This is the regression test for every later step.

### 1.9 Housekeeping
- `.gitignore`: add `refs/`, `*.pfm`, `/*.png` (renders in the repo root only, `scenes/*.png` stay tracked) and `__pycache__/`. Done.
- Fill the **CPU column** of the timing table (section 8). Done.

Done when: the timing line, PFM, CLI and all three scripts work, the references and noise floors are recorded, and the CPU timings are in `bench/results.csv`. PNG output is unchanged from before Phase 1 (byte compare).

---

## 5. Phase 2 — make the CPU code GPU-ready (plain C++)

Each step: change, rebuild, run the check, compile the shared headers with nvcc (below), benchmark when the machine is idle (CPU speed changes count as writeup material too), log the step in `PROGRESS.md`, then **one commit per step**.

**Benchmarks are interleaved same-session A/B (C11, C12):** keep each step's build in `out/check/bin/`, then run `bench.py --label "…" --base out/check/bin/<previous>.exe --base-commit <hash>`. It alternates the two builds run by run for each scene. Never compare against numbers from another day (the same code varied 6–10% between sessions), and never run two builds one after the other (the second loses up to 3% to heat).

**Device compile check (C8):** after every step, run `scripts/devcheck.py`. It compiles `tests/devcheck.cu`, a `__global__` function that calls `pathTrace`, with nvcc (`-arch=sm_86 --expt-relaxed-constexpr`), and lists every host-only function still called from device code. `--strict` fails on any of them. Warnings about host functions called from device code count as failures once the step that removes them is done. This turns "no host-only calls" into a test long before Phase 3.

**After 2.8 (C9):** once the new RNG passes against the current floor, re-measure the floor with it (`scripts/refs.py --floor`, about 10 min) so the GPU is compared against the same sampler. Also save a new exact baseline, since every image changes at 2.8. The references stay as they are.

**Check types:** **[exact]** = PFM files byte-identical to the image before the step, for the benchmark + correctness sets at low spp. **[stat]** = `compare.py` against the references, within the noise floor.

| Step | Change | Check |
|---|---|---|
| 2.1 | **`HD` macro** in a new `Common.h`: `#define HD __host__ __device__` when `__CUDACC__` is defined, otherwise empty. Put `HD inline` on every function `pathTrace` reaches. The other free functions in headers get plain `inline` (ODR fix). | [exact] |
| 2.2 | **No STL in the hot path:** `INF` → `FLT_MAX` (`<cfloat>`), `std::max` → small `HD inline` helpers that keep `std::max`'s exact behaviour (`a < b ? b : a`). `fmaxf` handles NaN differently and could change pixels. | [exact] |
| 2.3 | **Material indices:** flat `std::vector<Material>` in the scene. `Triangle`, `Sphere` and `Hit` hold `int matId`. The parser deduplicates materials with the same values. Quad-light materials added per D4. | [exact] |
| 2.4 | **`SceneView`:** plain struct of raw pointers + counts (tris, spheres, materials, quad lights, BVH nodes, prim indices, medium, camera, settings). Built from the vectors after parsing and the BVH build. Every function on the hot path takes `const SceneView&` and no longer takes `Scene`/`BVH`. | [exact] |
| 2.5 | **BVH reorder:** after the build, reorder the prims so leaves index them directly. Split spheres and triangles into one prim list with a type bit packed into the leaf, or two leaf kinds; decide while doing it. Remove `Prim` from traversal. The node layout stays the same for now (layout changes are Phase 4 and get measured). | [exact]\* |
| 2.6 | **Ray–box NaN:** make `AABB::hit` safe for axis-aligned rays (replace exact-zero direction components with a tiny signed value before inverting, so 0 × inf can't happen). | [exact] or [stat]\*\* |
| 2.7 | **Per-pixel function:** `renderPixel(const SceneView&, int x, int y, int s)` returns one sample's radiance. The caller adds it into the float accumulation buffer. Clamp/gamma/8-bit becomes `outputPass()`. The sum order per pixel stays `s = 0..spp-1`. The Whitted, analytic and direct integrators keep their own path in `render()`. | [exact] |
| 2.8 | **RNG:** stateless PCG. State = `{pixel, sample, dim}`. `next()` returns `pcgHash(pixel, sample, dim++)` turned into `[0,1)` (top 24 bits × 2⁻²⁴). Keep the "sample 0 at pixel centre" rule. Replaces every `std::mt19937`/`uniform_real_distribution` on the path tracer's path. The direct integrator can keep mt19937 (CPU only). | [stat] |
| 2.9 | **Bounce cap (D5)**, if agreed. | [exact] (expected) |

\* Exact unless two prims hit at the exact same `t`, where the visit order picks the winner. Any such pixels get investigated, not waved through.
\*\* If any pixels change, they're ones where the old code missed a box because of NaN. Then the step is a bug fix: record the changed pixel count in `PERF_LOG.md` and use [stat].

**After 2.8:** CPU and GPU will draw **the same random numbers**. Any difference then comes only from floating-point differences (nvcc FMA contraction, `sinf`/`powf`/`logf` precision) and the RR/scatter decisions they flip. That gives an extra GPU check besides the statistical one: most pixels should match closely, and a debug build with `--fmad=false` should match even more closely.

Done when: 2.1–2.8 (2.9) are done, every check has passed, the hot path has no STL and no host-only calls, and the post-Phase-2 CPU timings are recorded.

---

## 6. Phase 3 — simple CUDA kernel

### Build
- `CMakeLists.txt`: `include(CheckLanguage)` + `check_language(CUDA)`. If a CUDA compiler is found and `VPT_CUDA` is on: `enable_language(CUDA)`, `CMAKE_CUDA_ARCHITECTURES 86`, C++17 for CUDA, `--expt-relaxed-constexpr` for CUDA sources (needed by GLM), add `Kernels.cu`, define `VPT_HAS_CUDA`. OpenMP flags only on `.cpp` sources.
- With `-DVPT_CUDA=OFF`, or on a machine without CUDA, the build is exactly today's OpenMP renderer. Check that both configurations build.

### Files
- `CudaCheck.h`: `CUDA_CHECK(call)` macro that prints file, line and `cudaGetErrorString`, then exits. Wrap every runtime call. After each launch, `CUDA_CHECK(cudaGetLastError())`.
- `GpuRender.h`: host-facing declaration, no CUDA types: `void renderGPU(const SceneView& host, int spp, vec3* accumOut, GpuTimes* t);`
- `Kernels.cu`: device-side scene upload (one `cudaMalloc` + `cudaMemcpy` per array, then a device `SceneView` with device pointers, passed to the kernel by value), the kernel, the launch loop, freeing everything in one place.

### Kernel
- One thread per pixel, 2D grid, 16×16 blocks to start (Phase 4 sweeps this).
- **One launch per sample:** `renderKernel<<<grid, block>>>(view, s, accum)`, where each thread adds `renderPixel(view, x, y, s)` into `accum[y*w + x]`. No atomics are needed, since each pixel belongs to one thread.
- After the last sample: copy `accum` back, then **the same host `outputPass()` and PNG/PFM writers** as the CPU path.

### Timing
- `cudaFree(0)` before timing so context creation isn't counted.
- `cudaEvent` around the sample loop gives kernel time. Wall clock around upload + render + download gives end-to-end time. Report both.
- The same `time …` line plus `upload_ms` and `download_ms`, so `bench.py` handles both backends.

### Checks
- `compute-sanitizer --tool memcheck` (comes with the toolkit) on `cornellScatter` at 1–4 spp: no errors.
- [stat] against the references for the benchmark + correctness sets, within the noise floor.
- The close-match check from 2.8 (same random numbers).

Done when: all of the above pass and the **GPU column** of the timing table is filled in.

---

## 7. Phase 4 — profile and optimise

### Method
1. **Nsight Systems** on one full render per scene: where wall time goes (upload, kernels, gaps between launches).
2. **Nsight Compute** on one representative launch per scene (e.g. sample 8). Record: registers per thread, achieved occupancy, local memory traffic (spilling, the traversal stack), branch efficiency / active threads per warp, L1/L2 hit rates, and the top stall reasons.
3. Pick the biggest measured problem → **one change** → re-run `bench.py` + `ncu` + the [stat] check → write a `PERF_LOG.md` entry. Keep the change only if it helps and the check passes. A change that didn't help still gets a log entry.

### Candidates (measure first; order by what the profile shows)
- Register pressure from the size of `Hit` (shrinks a lot after 2.3).
- BVH node layout: 32-byte nodes, children stored as a pair (`left`, `left+1`), leaf count + offset in one int.
- SAH build (binned) instead of the median split, and up to 4 prims per leaf.
- Visit the nearer child first (by ray direction sign on the split axis). Traversal now pushes left then right in a fixed order.
- Where the traversal stack lives (local memory today), and whether a shorter stack fits in registers.
- Precomputed triangle edges `e1`, `e2` (recomputed on every test today), and sphere data trimmed to a 3×4 inverse.
- `pdfNEE` O(lights²) per vertex → reuse the `hitQuad` results inside `shadeMIS`.
- Warp divergence: surface vs volume events, Phong vs GGX.
- `__launch_bounds__`, `__ldg` / `const __restrict__` for scene data, block size sweep (8×8, 16×16, 32×8).
- `--use_fast_math` as a **separate, last** experiment, reported with its effect on image quality.

### `PERF_LOG.md` entry template
```
## <n>. <short title>
- metric before: <ncu/nsys numbers that motivated it>
- change: <what, which files, commit>
- result: <render ms per scene before → after, ncu metric before → after>
- image check: <compare.py result vs noise floor>
- kept: yes/no
```

---

## 8. Deliverables

- Code: one CMake project, shared CPU/GPU headers, `Kernels.cu`, and the harness (`compare.py`, `bench.py`, `check.py`).
- `bench/results.csv`: raw timing data.
- `PERF_LOG.md`: one entry per measured change.
- Timing table (goes into the writeup):

| Scene | Res | spp | CPU 20T (ms) | GPU v1 (ms) | GPU final (ms) | Speedup v1 / final | FLIP vs ref at 64 spp (CPU / GPU) |
|---|---|---|---|---|---|---|---|
| cornellScatter | 480² | 128 | 4060 (7.3 Ms/s) | | | | 0.177 / |
| hero | 800² | 128 | 10381 (7.9 Ms/s) | | | | 0.122 / |
| dragon | 640×480 | 128 | 12851 (3.1 Ms/s) | | | | 0.0224 / |
| mis | 768×512 | 128 | 5874 (8.6 Ms/s) | | | | 0.0110 / |

The final CPU column gets re-measured in the same session as the GPU numbers (C11). Today's values are the 2026-11-09 baseline. CPU column: Phase 1 build, i7-12700KF, 20 threads, median of 5 runs after 1 warm-up, render only (parse + BVH excluded; dragon adds 192 ms parse + 79 ms BVH). Run-to-run spread: dragon and mis within 1–2%; cornellScatter and hero up to 10%, likely from the mix of P and E cores. Raw rows in `bench/results.csv`. Ms/s = million samples (paths) per second. CPU FLIP is the 16-seed noise-floor mean.

- Writeup draft (markdown here, when asked): approach, harness and validation method, CPU vs GPU table, profile findings, the change-by-change log, image comparisons, what's left.

---

## 9. Risks and how to handle them

| Risk | Handling |
|---|---|
| A future CUDA or VS update breaks the host-compiler pairing | Resolved for now: CUDA 13.5 + VS 2026 verified. If an update breaks it, pin the working versions or use `-allow-unsupported-compiler` as a stopgap. |
| Duplicate symbols when a `.cu` includes the headers | Covered by 2.1 (`inline` on everything defined in headers). |
| Something in GLM won't compile under nvcc | GLM 1.0.0 supports CUDA. Work around just that function (e.g. `glm::inverse` stays host-only, in the parser); don't rewrite the maths. |
| **TDR** (Windows kills kernels running > ~2 s on the display GPU) | One launch per sample keeps launches short. If a single dragon sample gets close, split the launch into tiles. The D5 cap bounds the worst-case path. Don't change the TDR registry setting. |
| CPU timing noise (P/E cores, boost, thermals) | Warm-up + median of 5, record the thread count, nothing heavy running in the background, same power plan for every run. |
| Fireflies in `hero` make RMSE unstable | relMSE + FLIP on the clamped LDR image; the noise floor is measured per scene. |
| CPU vs GPU float differences look like bugs | After 2.8 the random numbers match, so any big difference in a pixel can be traced. `--fmad=false` build for debugging. |
| Scope creep in Phase 4 | Only change what a profile justifies. Each change is measured and logged, or it doesn't go in. |

---

## 10. New files (planned)

```
PLAN.md            this file
PROGRESS.md        progress log: bugs, plan changes, results, open items
PERF_LOG.md        phase 4 log
Common.h           HD macro, small maths helpers, FLT_MAX use
Image.h            pfm writer, output pass
Rng.h              pcg hash sampler (phase 2.8)
SceneView.h        raw-pointer scene view + builder (phase 2.4)
CudaCheck.h        CUDA_CHECK macro (phase 3)
GpuRender.h        host-facing gpu entry point (phase 3)
Kernels.cu         device upload, kernel, launch loop (phase 3)
scripts/vpt.py     shared helpers: scene sets, spp constants, pfm, metrics
scripts/compare.py rmse, relmse, flip vs reference
scripts/refs.py    reference renders + noise floor
scripts/bench.py   timing runs -> bench/results.csv
scripts/check.py   exact / statistical regression check
scripts/devcheck.py nvcc compile check of the shared headers
tests/devcheck.cu  kernel used by devcheck.py
bench/results.csv  raw timings
refs/              reference pfms (git-ignored)
```
