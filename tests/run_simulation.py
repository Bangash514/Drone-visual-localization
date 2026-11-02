# Date: 2026-11-02
# Change aim: run localization on a synthetic map at known positions, write pixel error and timings
# File: tests/run_simulation.py

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import localize  # noqa: E402
import simlib  # noqa: E402


def run_config(index, cases, world, size, mode, features, repeat, success_px):
    rows = []
    for c in cases:
        drone = simlib.make_drone(world, c["cx"], c["cy"], c["scale"], c["angle"], c["alpha"], c["beta"],
                                  size=size, seed=c["seed"])
        if mode == "known":
            scales = (1.0 / c["scale"],)
        elif mode == "search":
            scales = tuple(localize.DEFAULT_SCALES)
        else:
            scales = (1.0,)
        det, mat, hom, res = [], [], [], None
        for _ in range(repeat):
            res = index.localize(drone, scales, nfeatures=features)
            det.append(res["detect_ms"])
            mat.append(res["match_ms"])
            hom.append(res["homography_ms"])
        loc = res["loc"]
        err = None
        if loc is not None:
            err = math.hypot(float(loc["center"][0]) - c["cx"], float(loc["center"][1]) - c["cy"])
        d, m, h = statistics.median(det), statistics.median(mat), statistics.median(hom)
        rows.append({
            "case": c["id"], "scale": c["scale"], "angle": c["angle"], "alpha": c["alpha"], "beta": c["beta"],
            "cx": c["cx"], "cy": c["cy"], "matches": res["matches"],
            "inliers": loc["inliers"] if loc else 0, "flag_success": bool(loc and loc["success"]),
            "error_px": None if err is None else round(err, 3),
            "within_tol": bool(loc and loc["success"] and err <= success_px),
            "detect_ms": round(d, 1), "match_ms": round(m, 1), "homography_ms": round(h, 2),
            "total_query_ms": round(d + m + h, 1),
        })
    return rows


def summarize(rows):
    ok = [r for r in rows if r["flag_success"]]
    errs = [r["error_px"] for r in ok]
    tot = [r["total_query_ms"] for r in rows]
    return {
        "cases": len(rows),
        "success_rate_pct": round(100.0 * sum(r["within_tol"] for r in rows) / len(rows), 1),
        "flagged_success": len(ok),
        "mean_error_px": round(statistics.mean(errs), 2) if errs else None,
        "max_error_px": round(max(errs), 2) if errs else None,
        "median_detect_ms": round(statistics.median(r["detect_ms"] for r in rows), 1),
        "median_match_ms": round(statistics.median(r["match_ms"] for r in rows), 1),
        "median_total_query_ms": round(statistics.median(tot), 1),
        "max_total_query_ms": round(max(tot), 1),
        "cases_under_100ms_pct": round(100.0 * sum(t < 100.0 for t in tot) / len(tot), 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="sim_work")
    p.add_argument("--out", default=str(HERE.parent / "results"))
    p.add_argument("--map-w", type=int, default=7680)
    p.add_argument("--map-h", type=int, default=4320)
    p.add_argument("--cases", type=int, default=24)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--per-tile", type=int, nargs="+", default=[800])
    p.add_argument("--features", type=int, nargs="+", default=[1500])
    p.add_argument("--matchers", nargs="+", default=["bf", "flann"])
    p.add_argument("--modes", nargs="+", default=["none"], choices=["none", "known", "search"])
    p.add_argument("--success-px", type=float, default=5.0)
    p.add_argument("--tag", default="sim")
    a = p.parse_args()

    work, out = Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    size = (800, 500)
    world = simlib.make_map(a.map_w, a.map_h, seed=0)
    cases = simlib.make_cases(a.map_w, a.map_h, a.cases, seed=1, size=size)
    all_rows, summaries = [], []
    for per_tile in a.per_tile:
        idx_path = work / f"index_{per_tile}.npz"
        wp = work / "map.png"
        if not wp.exists():
            cv2.imwrite(str(wp), world)
        stats = localize.build_index(wp, idx_path, per_tile)
        for matcher in a.matchers:
            index = localize.MapIndex(idx_path, matcher)
            for features in a.features:
                for mode in a.modes:
                    rows = run_config(index, cases, world, size, mode, features, a.repeat, a.success_px)
                    for r in rows:
                        r.update({"matcher": matcher, "features": features, "per_tile": per_tile, "mode": mode})
                    s = summarize(rows)
                    s.update({"matcher": matcher, "features": features, "per_tile": per_tile, "mode": mode,
                              "index_keypoints": stats["n_keypoints"], "index_build_ms": stats["index_build_ms"],
                              "index_load_ms": round(index.load_ms, 1)})
                    summaries.append(s)
                    all_rows.extend(rows)
                    print(json.dumps(s), flush=True)
    with open(out / f"{a.tag}_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    meta = {"map_size": [a.map_w, a.map_h], "drone_size": list(size), "success_px": a.success_px,
            "cv2_python": cv2.__version__, "repeat": a.repeat, "cases": a.cases, "summaries": summaries}
    (out / f"{a.tag}_summary.json").write_text(json.dumps(meta, indent=2))
    cols = ["matcher", "mode", "features", "per_tile", "index_keypoints", "success_rate_pct", "mean_error_px",
            "max_error_px", "median_detect_ms", "median_match_ms", "median_total_query_ms",
            "max_total_query_ms", "cases_under_100ms_pct"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for s in summaries:
        lines.append("| " + " | ".join(str(s[c]) for c in cols) + " |")
    (out / f"{a.tag}_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
