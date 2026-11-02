# Date: 2026-11-02
# Change aim: plot per-case pixel error and query time from run_simulation.py output
# File: tests/plot_results.py

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--matcher", default="flann")
    p.add_argument("--features", type=int, default=1500)
    p.add_argument("--per-tile", type=int, default=800)
    p.add_argument("--mode", default="none")
    a = p.parse_args()
    rows = [r for r in csv.DictReader(open(a.csv))
            if r["matcher"] == a.matcher and int(r["features"]) == a.features
            and int(r["per_tile"]) == a.per_tile and r["mode"] == a.mode]
    err = [float(r["error_px"]) if r["error_px"] else float("nan") for r in rows]
    tot = [float(r["total_query_ms"]) for r in rows]
    scale = [float(r["scale"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    sc = ax[0].scatter(range(len(err)), err, c=scale, cmap="viridis")
    ax[0].set_xlabel("case")
    ax[0].set_ylabel("center error (map px)")
    ax[0].set_title("position error (SIMULATED data)")
    fig.colorbar(sc, ax=ax[0], label="drone px per map px")
    ax[1].bar(range(len(tot)), tot, color="#4c78a8")
    ax[1].axhline(100.0, color="red", linestyle="--", label="100 ms")
    ax[1].set_xlabel("case")
    ax[1].set_ylabel("total_query_ms")
    ax[1].set_title(f"{a.matcher}, {a.features} query kp, {a.per_tile} kp/tile")
    ax[1].legend()
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)


if __name__ == "__main__":
    main()
