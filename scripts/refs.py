"""refs.py - reference images and the noise floor for the statistical check.

    python scripts/refs.py               render missing references (refs/<scene>.pfm, 4096 spp)
    python scripts/refs.py --force       re-render all of them
    python scripts/refs.py --floor       measure the seed-to-seed error spread at the check spp

References use their own seed, so they share no samples with test renders. The noise floor renders
each scene with several seeds at the statistical-check spp and stores each seed's error against the
reference in refs/floor.json. refs/ is git-ignored; refs/README.txt records how each file was made.
"""
import argparse
import datetime
import json
import os
import tempfile
import time

import vpt


def renderRefs(exe, sel, force):
    os.makedirs(vpt.REFS, exist_ok=True)
    log = os.path.join(vpt.REFS, "README.txt")
    for name, path in sel:
        png = os.path.join(vpt.REFS, name + ".png")
        if os.path.isfile(vpt.pfmFor(png)) and not force:
            print(f"{name:15s} exists, skipped")
            continue
        t0 = time.time()
        vpt.render(exe, path, png, vpt.REF_SPP, seed=vpt.REF_SEED)
        secs = time.time() - t0
        print(f"{name:15s} {vpt.REF_SPP} spp in {secs:7.1f} s")
        with open(log, "a") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {name} spp={vpt.REF_SPP} "
                    f"seed={vpt.REF_SEED} secs={secs:.1f} commit={vpt.gitCommit()} exe={exe}\n")


def measureFloor(exe, sel):
    floor = {}
    if os.path.isfile(vpt.FLOOR_FILE):
        with open(vpt.FLOOR_FILE) as f:
            floor = json.load(f)
    keys = ["rmse", *vpt.METRICS]
    print(f"{len(vpt.FLOOR_SEEDS)} seeds at {vpt.STAT_SPP} spp, mean / std per metric\n")
    print(f"{'scene':15s} " + " ".join(f"{k:>20s}" for k in vpt.METRICS))
    with tempfile.TemporaryDirectory() as tmp:
        for name, path in sel:
            ref = vpt.readPFM(os.path.join(vpt.REFS, name + ".pfm"))
            gamma = vpt.sceneGamma(path)
            per = {k: [] for k in keys}
            for seed in vpt.FLOOR_SEEDS:
                out = os.path.join(tmp, f"{name}_{seed}.png")
                vpt.render(exe, path, out, vpt.STAT_SPP, seed=seed)
                m = vpt.metrics(vpt.readPFM(vpt.pfmFor(out)), ref, gamma)
                for k in keys:
                    per[k].append(m[k])
            floor[name] = {"spp": vpt.STAT_SPP, "seeds": vpt.FLOOR_SEEDS, "commit": vpt.gitCommit(), **per}
            print(f"{name:15s} " + " ".join(f"{_ms(per[k]):>20s}" for k in vpt.METRICS))
    with open(vpt.FLOOR_FILE, "w") as f:
        json.dump(floor, f, indent=1)
    print(f"\nwrote {os.path.relpath(vpt.FLOOR_FILE, vpt.ROOT)}")


def _ms(s):
    import statistics
    return f"{statistics.mean(s):.3g} / {statistics.stdev(s):.2g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", help="renderer (default $VPT_EXE or the newest build under out/build)")
    ap.add_argument("--scenes", nargs="+", help="subset of the scenes")
    ap.add_argument("--force", action="store_true", help="re-render existing references")
    ap.add_argument("--floor", action="store_true", help="measure the noise floor instead")
    a = ap.parse_args()

    exe = vpt.findExe(a.exe)
    sel = vpt.scenes("all", a.scenes)
    if a.floor:
        measureFloor(exe, sel)
    else:
        renderRefs(exe, sel, a.force)


if __name__ == "__main__":
    main()
