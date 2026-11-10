"""compare.py - image error of a test PFM against a reference PFM.

    python scripts/compare.py test.pfm ref.pfm [--gamma G | --scene file.test] [--flipmap map.ppm]

Prints rmse and relmse (linear values), mean LDR-FLIP (after the renderer's clamp and gamma), and
blockrel and bias, the relative error of 16x16 block means (noise averages out, bias doesn't).
GPU and CPU renders use different random numbers, so judge results against the noise floor
(scripts/refs.py --floor), never pixel by pixel.
"""
import argparse

import vpt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test")
    ap.add_argument("ref")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gamma", type=float, help="output gamma (default 1)")
    g.add_argument("--scene", help="read the gamma from this scene file")
    ap.add_argument("--flipmap", help="write the FLIP error map (magma) as a .ppm")
    a = ap.parse_args()

    gamma = vpt.sceneGamma(a.scene) if a.scene else (a.gamma or 1.0)
    m = vpt.metrics(vpt.readPFM(a.test), vpt.readPFM(a.ref), gamma, flipMap=bool(a.flipmap))
    if a.flipmap:
        vpt.writePPM(a.flipmap, m["flipmap"])
    print(f"rmse={m['rmse']:.6g} relmse={m['relmse']:.6g} flip={m['flip']:.6g} "
          f"blockrel={m['blockrel']:.6g} bias={100 * m['bias']:+.3f}% nonfinite={m['nonfinite']}")


if __name__ == "__main__":
    main()
