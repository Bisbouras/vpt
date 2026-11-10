"""bench.py - timed renders of the benchmark set, appended to bench/results.csv.

    python scripts/bench.py [--runs 5] [--warmup 1] [--spp 128] [--scenes hero dragon] [--label text]

Each scene gets warm-up runs, then timed runs. The median of the timed runs is recorded, plus the
min and max render time so noisy runs show up.
"""
import argparse
import csv
import datetime
import os
import statistics
import tempfile

import vpt

FIELDS = ["date", "commit", "backend", "label", "scene", "w", "h", "spp", "threads", "runs",
          "parse_ms", "bvh_ms", "render_ms", "render_min_ms", "render_max_ms", "msamples_s"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", help="renderer (default $VPT_EXE or the newest build under out/build)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--spp", type=int, default=vpt.BENCH_SPP)
    ap.add_argument("--scenes", nargs="+", help="subset of the benchmark set")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--label", default="", help="free text stored with the rows, e.g. the change being measured")
    ap.add_argument("--extra", default="", help="extra renderer arguments, e.g. --gpu")
    ap.add_argument("--csv", default=os.path.join(vpt.ROOT, "bench", "results.csv"))
    a = ap.parse_args()

    exe = vpt.findExe(a.exe)
    commit = vpt.gitCommit()
    print(f"renderer {exe}\ncommit {commit}, {a.warmup} warm-up + {a.runs} timed runs, {a.spp} spp\n")
    print(f"{'scene':15s} {'render ms':>10s} {'min':>9s} {'max':>9s} {'Msamples/s':>11s}")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, path in vpt.scenes("bench", a.scenes):
            out = os.path.join(tmp, name + ".png")
            for _ in range(a.warmup):
                vpt.render(exe, path, out, a.spp, extra=a.extra.split())
            times = [vpt.render(exe, path, out, a.spp, extra=a.extra.split()) for _ in range(a.runs)]
            w, h = vpt.sceneSize(path)
            med = {k: statistics.median(t[k] for t in times) for k in ("parse_ms", "bvh_ms", "render_ms")}
            rs = [t["render_ms"] for t in times]
            msamples = w * h * a.spp / (med["render_ms"] / 1000.0) / 1e6
            rows.append({
                "date": datetime.datetime.now().isoformat(timespec="seconds"), "commit": commit,
                "backend": a.backend, "label": a.label, "scene": name, "w": w, "h": h, "spp": a.spp,
                "threads": int(times[0].get("threads", 0)), "runs": a.runs,
                "parse_ms": f"{med['parse_ms']:.1f}", "bvh_ms": f"{med['bvh_ms']:.1f}",
                "render_ms": f"{med['render_ms']:.1f}", "render_min_ms": f"{min(rs):.1f}",
                "render_max_ms": f"{max(rs):.1f}", "msamples_s": f"{msamples:.3f}",
            })
            print(f"{name:15s} {med['render_ms']:10.1f} {min(rs):9.1f} {max(rs):9.1f} {msamples:11.3f}")

    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    new = not os.path.isfile(a.csv)
    with open(a.csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        wr.writerows(rows)
    print(f"\nappended {len(rows)} rows to {os.path.relpath(a.csv, vpt.ROOT)}")


if __name__ == "__main__":
    main()
