"""check.py - regression check over the benchmark and correctness scenes.

    python scripts/check.py --save-baseline DIR   render every scene at 8 spp into DIR
    python scripts/check.py --exact DIR           re-render and byte-compare png + pfm with DIR
    python scripts/check.py --stat                64 spp vs refs/, within the noise floor

--exact is for changes that must not alter any output (Phase 2 refactors). --stat is for changes that
alter the random numbers or the device (Phase 2 RNG, GPU). Prints one line per scene and exits
non-zero if any scene fails. Extra renderer arguments (e.g. --gpu) go in --extra.
"""
import argparse
import filecmp
import json
import os
import sys
import tempfile

import vpt


def runExact(exe, sel, base, extra, save):
    if save:
        os.makedirs(base, exist_ok=True)
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, path in sel:
            outDir = base if save else tmp
            png = os.path.join(outDir, name + ".png")
            vpt.render(exe, path, png, vpt.EXACT_SPP, extra=extra)
            if save:
                print(f"{name:15s} saved")
                continue
            ref = os.path.join(base, name + ".png")
            if not os.path.isfile(ref):
                sys.exit(f"no baseline for {name} in {base}")
            same = [filecmp.cmp(p, r, shallow=False)
                    for p, r in ((png, ref), (vpt.pfmFor(png), vpt.pfmFor(ref)))]
            ok = all(same)
            fails += not ok
            print(f"{name:15s} {'PASS' if ok else 'FAIL'}  png {'same' if same[0] else 'DIFFERS'}, "
                  f"pfm {'same' if same[1] else 'DIFFERS'}")
    return fails


def runStat(exe, sel, seed, extra):
    if not os.path.isfile(vpt.FLOOR_FILE):
        sys.exit("no refs/floor.json, run scripts/refs.py then scripts/refs.py --floor")
    with open(vpt.FLOOR_FILE) as f:
        floor = json.load(f)
    fails = 0
    lims = ", ".join(f"{k} {'|z|' if two else 'z'} <= {lim:g}" for k, (two, lim) in vpt.METRICS.items())
    print(f"z = distance from the noise-floor mean in floor std. passes when {lims}. ! marks a failure\n")
    print(f"{'scene':15s} " + " ".join(f"{k:>9s}" for k in vpt.METRICS) + f" {'bias %':>8s}")
    with tempfile.TemporaryDirectory() as tmp:
        for name, path in sel:
            if name not in floor or "bias" not in floor[name]:
                sys.exit(f"no current noise floor for {name}, run scripts/refs.py --floor")
            fl = floor[name]
            png = os.path.join(tmp, name + ".png")
            vpt.render(exe, path, png, fl["spp"], seed=seed, extra=extra)
            m = vpt.metrics(vpt.readPFM(vpt.pfmFor(png)), vpt.readPFM(os.path.join(vpt.REFS, name + ".pfm")),
                            vpt.sceneGamma(path))
            res, ok = vpt.judge(m, fl)
            fails += not ok
            cells = [f"{z:+8.1f}{' ' if good else '!'}" for z, good in res.values()]
            note = f"  {m['nonfinite']} non-finite pixels" if m["nonfinite"] else ""
            print(f"{name:15s} " + " ".join(cells) + f" {100 * m['bias']:+8.2f}  {'PASS' if ok else 'FAIL'}{note}")
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--save-baseline", metavar="DIR")
    mode.add_argument("--exact", metavar="DIR")
    mode.add_argument("--stat", action="store_true")
    ap.add_argument("--exe", help="renderer (default $VPT_EXE or the newest build under out/build)")
    ap.add_argument("--scenes", nargs="+", help="subset of the scenes")
    ap.add_argument("--seed", type=int, default=0, help="seed for --stat (default 0)")
    ap.add_argument("--extra", default="", help="extra renderer arguments, e.g. --gpu")
    a = ap.parse_args()

    exe = vpt.findExe(a.exe)
    sel = vpt.scenes("all", a.scenes)
    extra = a.extra.split()
    print(f"renderer {exe}\n")
    if a.stat:
        fails = runStat(exe, sel, a.seed, extra)
    else:
        fails = runExact(exe, sel, a.save_baseline or a.exact, extra, save=bool(a.save_baseline))
    if not a.save_baseline:
        print(f"\n{len(sel) - fails}/{len(sel)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
