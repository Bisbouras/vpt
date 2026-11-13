"""vpt.py - shared helpers for the benchmark, comparison and check scripts.

Scene sets, renderer lookup and invocation, PFM reading, and image metrics.
Only numpy and flip-evaluator are needed.
"""
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, "refs")

# spp per use - bench: timed runs, stat: statistical check, exact: byte check, ref: references
BENCH_SPP = 128
STAT_SPP = 64
EXACT_SPP = 8
REF_SPP = 4096

# seeds - references use their own seed so they share no samples with test renders
REF_SEED = 1000
FLOOR_SEEDS = list(range(16))
FLOOR_FILE = os.path.join(REFS, "floor.json")
# relmse ignores this fraction of worst pixels, block metrics this fraction of worst blocks (fireflies)
RELMSE_TRIM = 0.001
BLOCK_TRIM = 0.01
# block metrics average BLOCK x BLOCK pixels, which removes most noise but keeps bias
BLOCK = 16

# stat pass rule - metric name -> (two-sided, K): within K std of the cpu seed-to-seed spread.
# bias can go either way, the rest only get worse upwards. flip has heavier tails, so a wider K.
METRICS = {"relmse": (False, 4.0), "flip": (False, 5.0), "blockrel": (False, 4.0), "bias": (True, 4.0)}

# bench - timed set, check - extra code paths (nee on, cosine, absorption only, transformed spheres)
SCENES = [
    ("cornellScatter", "scenes/cornellScatter.test", "bench"),
    ("hero",           "scenes/hero.test",           "bench"),
    ("dragon",         "scenes/dragon.test",         "bench"),
    ("mis",            "scenes/mis.test",            "bench"),
    ("cornellBRDF",    "scenes/cornellBRDF.test",    "check"),
    ("cornellCosine",  "scenes/cornellCosine.test",  "check"),
    ("cornellFog",     "scenes/cornellFog.test",     "check"),
    ("ggx",            "scenes/ggx.test",            "check"),
]

EXE_CANDIDATES = [
    "out/build/x64-Release/raytracer.exe",
    "out/build/cli-release/raytracer.exe",
]


def scenes(which="all", only=None):
    """(name, absolute path) for the bench set, check set or both, optionally filtered by name."""
    out = []
    for name, rel, role in SCENES:
        if which != "all" and role != which:
            continue
        if only and name not in only:
            continue
        out.append((name, os.path.join(ROOT, rel)))
    if only:
        missing = set(only) - {n for n, _ in out}
        if missing:
            sys.exit(f"unknown scene(s): {', '.join(sorted(missing))}")
    return out


def findExe(exe=None):
    """the renderer to run - explicit path, $VPT_EXE, or the newest known build."""
    exe = exe or os.environ.get("VPT_EXE")
    if exe:
        if not os.path.isfile(exe):
            sys.exit(f"renderer not found: {exe}")
        return os.path.abspath(exe)
    found = [os.path.normpath(os.path.join(ROOT, c)) for c in EXE_CANDIDATES
             if os.path.isfile(os.path.join(ROOT, c))]
    if not found:
        sys.exit("no renderer build found, pass --exe or set VPT_EXE")
    return max(found, key=os.path.getmtime)


def sceneGamma(path):
    """the scene's output gamma (last 'gamma' line wins, like the parser)."""
    g = 1.0
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "gamma":
                g = float(parts[1])
    return g


def sceneSize(path):
    """(w, h) from the scene's 'size' line (parser default 640x480)."""
    w, h = 640, 480
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "size":
                w, h = int(parts[1]), int(parts[2])
    return w, h


def judge(m, floor):
    """per-metric (z, ok) against the noise floor, plus overall pass.

    z is the distance from the floor mean in floor standard deviations. A metric passes when
    z <= K, or |z| <= K for the two-sided bias.
    """
    res = {}
    for k, (two, lim) in METRICS.items():
        s = np.asarray(floor[k], dtype=np.float64)
        z = float((m[k] - s.mean()) / s.std(ddof=1))
        res[k] = (z, abs(z) <= lim if two else z <= lim)
    ok = m["nonfinite"] == 0 and all(good for _, good in res.values())
    return res, ok


TIME_RE = re.compile(r"^time (.*)$", re.M)


def render(exe, scene, outPng, spp, seed=0, extra=()):
    """runs the renderer quietly and returns the parsed timing line as a dict of floats."""
    cmd = [exe, scene, "--spp", str(spp), "--seed", str(seed), "--out", outPng, "--quiet", *extra]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"renderer failed ({res.returncode}): {' '.join(cmd)}\n{res.stdout}{res.stderr}")
    m = TIME_RE.search(res.stdout)
    if not m:
        sys.exit(f"no timing line in renderer output (old build?): {exe}\n{res.stdout}")
    return {k: float(v) for k, v in (kv.split("=") for kv in m.group(1).split())}


def pfmFor(png):
    return os.path.splitext(png)[0] + ".pfm"


def readPFM(path):
    """HxWx3 float32, top row first (pfm stores rows bottom to top)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"PF":
            raise ValueError(f"{path}: not a colour pfm")
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.fromfile(f, dtype="<f4" if scale < 0 else ">f4")
    if data.size != w * h * 3:
        raise ValueError(f"{path}: expected {w * h * 3} floats, got {data.size}")
    return np.flipud(data.reshape(h, w, 3)).astype(np.float32)


def toLDR(img, gamma):
    """the renderer's own output mapping, without the 8-bit rounding."""
    c = np.clip(img, 0.0, 1.0)
    return np.power(c, 1.0 / gamma).astype(np.float32) if gamma != 1.0 else c


def metrics(test, ref, gamma, flipMap=False):
    """rmse and relmse (worst 0.1% pixels dropped) on linear values, mean ldr-flip on the display-mapped images."""
    if test.shape != ref.shape:
        raise ValueError(f"size mismatch {test.shape} vs {ref.shape}")
    bad = int((~np.isfinite(test)).sum())
    t = np.nan_to_num(test, nan=0.0, posinf=0.0, neginf=0.0)
    d2 = (t - ref) ** 2
    # relmse - per-pixel, worst RELMSE_TRIM of pixels dropped so single fireflies don't dominate
    rel = (d2 / (ref * ref + 1e-2)).mean(axis=2).ravel()
    keep = max(1, int(round(rel.size * (1.0 - RELMSE_TRIM))))
    out = {
        "rmse": float(np.sqrt(d2.mean())),
        "relmse": float(np.partition(rel, keep - 1)[:keep].mean()),
        "nonfinite": bad,
    }
    out.update(blockMetrics(t, ref))
    import flip_evaluator as flip
    errMap, meanFlip, _ = flip.evaluate(toLDR(ref, gamma), toLDR(t, gamma), "LDR", applyMagma=flipMap)
    out["flip"] = float(meanFlip)
    if flipMap:
        out["flipmap"] = errMap
    return out


def blockMetrics(test, ref):
    """bias (signed mean) and blockrel (mean square) of the relative error of BLOCK x BLOCK means."""
    h, w = (test.shape[0] // BLOCK) * BLOCK, (test.shape[1] // BLOCK) * BLOCK

    def means(img):
        return img[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK, 3).mean(axis=(1, 3))

    tb, rb = means(test), means(ref)
    e = ((tb - rb) / (rb + 1e-2)).mean(axis=2).ravel()
    keep = np.argsort(np.abs(e))[:max(1, int(round(e.size * (1.0 - BLOCK_TRIM))))]
    e = e[keep]
    return {"bias": float(e.mean()), "blockrel": float((e * e).mean())}


def writePPM(path, img):
    """8-bit binary ppm from an HxWx3 [0,1] float image."""
    h, w, _ = img.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())


def gitCommit():
    """short head hash, with +dirty when renderer source differs from it (docs and bench data don't count)."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "*.h", "*.cpp", "*.cu", "CMakeLists.txt"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
