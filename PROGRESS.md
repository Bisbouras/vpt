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
| `hero`'s statistical check only catches errors of about 5% or more (fireflies survive in its reference). Its code paths are covered more tightly by `cornellScatter`. | 2026-11-09 | Accepted limit |
| CPU timings for `cornellScatter` and `hero` vary up to 10% between runs (P/E cores). Medians of 5 are used, and min/max are recorded. | 2026-11-09 | Keep in mind when reading small CPU changes |

## Bug register

| ID | Bug | Found | Status |
|---|---|---|---|
| B1 | `done++` in `render()` is a data race under OpenMP. It also printed progress lines from several threads. | Outline (known) | **Fixed** in Phase 1 (`ae82772`): `std::atomic<int>`, printing off with `--quiet` |
| B2 | `AABB::hit` can produce NaN for axis-aligned rays (0 × inf). CPU and GPU min/max treat NaN differently. | Outline (known) | Open → Phase 2.6 |
| B3 | Functions defined in headers aren't `inline`. Linking will fail with duplicate symbols as soon as a second file (`Kernels.cu`) includes them. Latent today. | Phase 0.5 code read | **Fixed** in 2.1 |
| B4 | No bounce cap for `maxdepth -1`: only Russian roulette ends a path. Fine on the CPU, but on the GPU it means long-tailed warps and a risk of hitting the Windows 2 s kernel timeout. | Phase 0.5 code read | Open → Phase 2.9 (decision D5) |

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

  **Result (withdrawn, see 2.2):** this A/B ran the builds one after the other, with Phase 1 second, and 2.2 showed that the second build in such a run comes out slower. So the "2.1 is about 2% faster" reading is not trustworthy. An interleaved re-run is needed before any 2.1 speed-up can be claimed. Most of the gap to the 2026-11-09 baseline is the machine: the identical Phase 1 code runs 6–10% faster today. Timings from different sessions can't be compared (C11).
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
- **Harness:** `bench.py --base EXE --base-commit C` times a base build interleaved with the build under test, alternating the order of each pair, and prints the change. The 16 earlier rows from back-to-back A/B runs keep their numbers but are labelled `[sequential A/B, order-biased]` in `bench/results.csv`. Builds for A/B are kept in `out/check/bin/` (`8ff144a.exe` = 2.1, `2.2.exe`).
