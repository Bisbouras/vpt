"""devcheck.py - compiles tests/devcheck.cu with nvcc to check the shared headers build for the gpu.

    python scripts/devcheck.py            report errors and host calls left in device code
    python scripts/devcheck.py --strict   also fail on any host call (the Phase 2 end state)

The kernel calls pathTrace, so every function the path tracer reaches is compiled for the device.
Calls from device code to host-only functions (std::, STL containers, mt19937) are reported by name.
They are expected until Phase 2 removes them. The full nvcc log goes to out/check/devcheck.log.
"""
import argparse
import collections
import os
import re
import subprocess
import sys
import tempfile

import vpt

NVCC_FLAGS = ["-std=c++17", "-arch=sm_86", "--expt-relaxed-constexpr", "-c"]
HOST_CALL = re.compile(r'calling a (?:constexpr )?__host__ function\("([^"]*)"\) from a __host__ __device__ function\("([^"]*)"\)')
DIAG = re.compile(r"(warning|error) #?(\w+)(?:-D)?:")


def vcvars():
    """vcvars64.bat of the newest visual studio, via vswhere."""
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(vswhere):
        sys.exit("vswhere.exe not found, is visual studio installed?")
    vs = subprocess.run([vswhere, "-latest", "-property", "installationPath"],
                        capture_output=True, text=True, check=True).stdout.strip()
    bat = os.path.join(vs, "VC", "Auxiliary", "Build", "vcvars64.bat")
    if not os.path.isfile(bat):
        sys.exit(f"vcvars64.bat not found under {vs}")
    return bat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="fail on host calls from device code too")
    a = ap.parse_args()

    src = os.path.join(vpt.ROOT, "tests", "devcheck.cu")
    with tempfile.TemporaryDirectory() as tmp:
        obj = os.path.join(tmp, "devcheck.obj")
        nvcc = " ".join(["nvcc", *NVCC_FLAGS, f'-I"{os.path.join(vpt.ROOT, "include")}"',
                         f'-I"{vpt.ROOT}"', f'"{src}"', "-o", f'"{obj}"'])
        res = subprocess.run(f'cmd /c ""{vcvars()}" >nul 2>nul && {nvcc}"', shell=True,
                             capture_output=True, text=True)
    log = res.stdout + res.stderr
    os.makedirs(os.path.join(vpt.ROOT, "out", "check"), exist_ok=True)
    with open(os.path.join(vpt.ROOT, "out", "check", "devcheck.log"), "w") as f:
        f.write(log)

    diags = collections.Counter(f"{k} {n}" for k, n in DIAG.findall(log))
    calls = collections.Counter(m.group(1) for m in HOST_CALL.finditer(log))
    callers = collections.defaultdict(set)
    for m in HOST_CALL.finditer(log):
        callers[m.group(1)].add(m.group(2).split("(")[0])

    errors = res.returncode != 0
    print(f"nvcc {'FAILED' if errors else 'compiled'} ({' '.join(NVCC_FLAGS)})")
    print("diagnostics: " + (", ".join(f"{n} x{c}" for n, c in sorted(diags.items())) or "none"))
    print(f"host functions called from device code: {len(calls)} distinct")
    for fn, c in sorted(calls.items(), key=lambda kv: -kv[1]):
        print(f"  {c:4d}  {fn}   <- {', '.join(sorted(callers[fn]))}")
    if errors:
        print("\nfirst errors:")
        for line in [l for l in log.splitlines() if "error" in l][:10]:
            print("  " + line.strip())

    ok = not errors and not (a.strict and calls)
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
