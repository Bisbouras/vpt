"""bench.py - timed renders of the benchmark set, appended to bench/results.csv.

    python scripts/bench.py [--runs 5] [--warmup 1] [--spp 128] [--scenes hero dragon] [--label text]
    python scripts/bench.py --label "2.3 ..." --base out/check/bin/abc123.exe --base-commit abc123

Each scene gets warm-up runs, then timed runs. The median of the timed runs is recorded, plus the
min and max render time so noisy runs show up.

With --base, the base build and the build under test are timed interleaved per scene, in the order
B T T B B T ..., so a machine that slows down as it heats up affects both equally. Both get rows
in the CSV, and the change in median render time is printed.
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


def row(name, path, times, commit, label, a):
    w, h = vpt.sceneSize(path)
    med = {k: statistics.median(t[k] for t in times) for k in ("parse_ms", "bvh_ms", "render_ms")}
    rs = [t["render_ms"] for t in times]
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"), "commit": commit,
        "backend": a.backend, "label": label, "scene": name, "w": w, "h": h, "spp": a.spp,
        "threads": int(times[0].get("threads", 0)), "runs": len(times),
        "parse_ms": f"{med['parse_ms']:.1f}", "bvh_ms": f"{med['bvh_ms']:.1f}",
        "render_ms": f"{med['render_ms']:.1f}", "render_min_ms": f"{min(rs):.1f}",
        "render_max_ms": f"{max(rs):.1f}", "msamples_s": f"{w * h * a.spp / med['render_ms'] / 1e3:.3f}",
    }


def show(tag, r):
    print(f"{r['scene']:15s} {tag:5s} {float(r['render_ms']):10.1f} {float(r['render_min_ms']):9.1f} "
          f"{float(r['render_max_ms']):9.1f} {float(r['msamples_s']):11.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", help="renderer under test (default $VPT_EXE or the newest build under out/build)")
    ap.add_argument("--commit", help="commit the renderer was built from, when it isn't HEAD")
    ap.add_argument("--label", default="", help="free text stored with the rows, e.g. the change being measured")
    ap.add_argument("--base", help="base build to time interleaved with the one under test")
    ap.add_argument("--base-commit", help="commit the base build was made from")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--spp", type=int, default=vpt.BENCH_SPP)
    ap.add_argument("--scenes", nargs="+", help="subset of the benchmark set")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--extra", default="", help="extra renderer arguments, e.g. --gpu")
    ap.add_argument("--csv", default=os.path.join(vpt.ROOT, "bench", "results.csv"))
    a = ap.parse_args()

    exe = vpt.findExe(a.exe)
    commit = a.commit or vpt.gitCommit()
    base = vpt.findExe(a.base) if a.base else None
    if base and not a.base_commit:
        ap.error("--base needs --base-commit")
    print(f"test {exe} ({commit})")
    if base:
        print(f"base {base} ({a.base_commit}), interleaved")
    print(f"{a.warmup} warm-up + {a.runs} timed runs each, {a.spp} spp\n")
    print(f"{'scene':15s} {'':5s} {'render ms':>10s} {'min':>9s} {'max':>9s} {'Msamples/s':>11s}")

    extra = a.extra.split()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, path in vpt.scenes("bench", a.scenes):
            out = os.path.join(tmp, name + ".png")
            if not base:
                for _ in range(a.warmup):
                    vpt.render(exe, path, out, a.spp, extra=extra)
                times = [vpt.render(exe, path, out, a.spp, extra=extra) for _ in range(a.runs)]
                rows.append(row(name, path, times, commit, a.label, a))
                show("", rows[-1])
                continue
            # interleaved - pairs alternate order (BT, TB, BT ...) to cancel drift
            for _ in range(a.warmup):
                vpt.render(base, path, out, a.spp, extra=extra)
                vpt.render(exe, path, out, a.spp, extra=extra)
            tb, tt = [], []
            for i in range(a.runs):
                order = [(base, tb), (exe, tt)] if i % 2 == 0 else [(exe, tt), (base, tb)]
                for e, acc in order:
                    acc.append(vpt.render(e, path, out, a.spp, extra=extra))
            rb = row(name, path, tb, a.base_commit, f"base for: {a.label}", a)
            rt = row(name, path, tt, commit, a.label, a)
            rows += [rb, rt]
            show("base", rb)
            show("test", rt)
            d = (float(rt["render_ms"]) / float(rb["render_ms"]) - 1) * 100
            print(f"{'':15s} {'':5s} {d:+9.1f}%")

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
