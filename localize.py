#!/usr/bin/env python3
"""
Drone-to-map localization.

Locate a small aerial (drone) image inside a large map image by matching local
features and estimating a homography.

Two matchers are supported:
  * superglue : SuperPoint + SuperGlue (learned, most robust, slow on CPU)
  * orb       : ORB + brute-force Hamming matching (classical, fast, less robust)

Example
-------
python localize.py --drone data/crop_drone.jpg --map data/full_image.jpg \
    --out results --method superglue --superglue-path ../SuperGluePretrainedNetwork \
    --map-max-side 2000 --max-keypoints 512 --sinkhorn-iters 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resize_max_side(img, max_side):
    """Downscale so the longest side is <= max_side. Returns (image, scale)."""
    if not max_side:
        return img, 1.0
    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale == 1.0:
        return img, 1.0
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), scale


def empty_matches():
    return (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32),
            np.zeros((0,), np.float32))


# --------------------------------------------------------------------------- #
# Matchers: each returns run(gray0, gray1) -> (pts0, pts1, confidence)
# --------------------------------------------------------------------------- #
def build_superglue(args):
    import torch

    sg_path = Path(args.superglue_path).expanduser().resolve()
    if not (sg_path / "models" / "matching.py").exists():
        sys.exit(f"SuperGlue repo not found at {sg_path}\n"
                 "Clone https://github.com/magicleap/SuperGluePretrainedNetwork "
                 "and pass --superglue-path.")
    sys.path.insert(0, str(sg_path))
    from models.matching import Matching  # noqa: E402

    torch.set_grad_enabled(False)
    config = {
        "superpoint": {
            "nms_radius": args.nms_radius,
            "keypoint_threshold": args.keypoint_threshold,
            "max_keypoints": args.max_keypoints,
        },
        "superglue": {
            "weights": args.superglue_weights,
            "sinkhorn_iterations": args.sinkhorn_iters,
            "match_threshold": args.match_threshold,
        },
    }
    model = Matching(config).eval().to(args.device)

    def to_tensor(gray):
        arr = gray.astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None, None].to(args.device)

    def run(g0, g1):
        pred = model({"image0": to_tensor(g0), "image1": to_tensor(g1)})
        k0 = pred["keypoints0"][0].cpu().numpy()
        k1 = pred["keypoints1"][0].cpu().numpy()
        m = pred["matches0"][0].cpu().numpy()
        c = pred["matching_scores0"][0].cpu().numpy()
        valid = m > -1
        return k0[valid], k1[m[valid]], c[valid]

    return run


def build_orb(args):
    orb = cv2.ORB_create(nfeatures=args.max_keypoints)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def run(g0, g1):
        k0, d0 = orb.detectAndCompute(g0, None)
        k1, d1 = orb.detectAndCompute(g1, None)
        if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2:
            return empty_matches()
        good = []
        for pair in bf.knnMatch(d0, d1, k=2):
            if len(pair) == 2 and pair[0].distance < args.ratio * pair[1].distance:
                good.append(pair[0])
        if not good:
            return empty_matches()
        p0 = np.float32([k0[m.queryIdx].pt for m in good])
        p1 = np.float32([k1[m.trainIdx].pt for m in good])
        conf = np.float32([1.0 - m.distance / 256.0 for m in good])
        return p0, p1, conf

    return run


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def estimate_location(p0, p1, drone_shape, ransac_thresh):
    """Homography drone->map. Returns dict (or None) in *resized map* pixels."""
    if len(p0) < 4:
        return None
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, ransac_thresh)
    if H is None:
        return None
    h, w = drone_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    return {
        "H": H,
        "inlier_mask": mask.ravel().astype(bool),
        "quad": quad,
        "center": quad.mean(axis=0),
        # A sane footprint is a convex quadrilateral; a twisted one means a bad fit.
        "quad_convex": bool(cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2))),
    }


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def draw_matches(g0, g1, p0, p1, inliers, max_lines=150, height=700):
    s0, s1 = height / g0.shape[0], height / g1.shape[0]
    a = cv2.cvtColor(cv2.resize(g0, None, fx=s0, fy=s0), cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(cv2.resize(g1, None, fx=s1, fy=s1), cv2.COLOR_GRAY2BGR)
    canvas = cv2.hconcat([a, b])
    order = np.argsort(~inliers)  # inliers first
    for i in order[:max_lines]:
        x0, y0 = p0[i] * s0
        x1, y1 = p1[i] * s1
        color = (0, 220, 0) if inliers[i] else (0, 0, 220)
        cv2.line(canvas, (int(x0), int(y0)), (int(x1 + a.shape[1]), int(y1)), color, 1, cv2.LINE_AA)
    return canvas


def draw_overlay(map_gray, loc, label):
    overlay = cv2.cvtColor(map_gray, cv2.COLOR_GRAY2BGR)
    quad = loc["quad"].astype(np.int32).reshape(-1, 1, 2)
    thickness = max(2, overlay.shape[1] // 600)
    cv2.polylines(overlay, [quad], True, (0, 255, 0), thickness, cv2.LINE_AA)
    cx, cy = loc["center"].astype(int)
    cv2.circle(overlay, (cx, cy), thickness * 3, (0, 0, 255), -1)
    cv2.putText(overlay, label, (cx + 12, cy - 12), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, overlay.shape[1] / 2500), (0, 255, 0), thickness, cv2.LINE_AA)
    return overlay


def zoom_crop(overlay, quad, margin=0.5):
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    H, W = overlay.shape[:2]
    x0, x1 = int(max(0, x0 - mx)), int(min(W, x1 + mx))
    y0, y1 = int(max(0, y0 - my)), int(min(H, y1 + my))
    return overlay[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drone", required=True, help="path to the drone / query image")
    p.add_argument("--map", required=True, help="path to the large map image")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--method", choices=["superglue", "orb"], default="superglue")
    p.add_argument("--device", default="cpu", help="cpu or cuda (superglue only)")

    g = p.add_argument_group("image size")
    g.add_argument("--drone-max-side", type=int, default=640,
                   help="downscale drone image so its longest side <= this (0 = off)")
    g.add_argument("--map-max-side", type=int, default=2000,
                   help="downscale map so its longest side <= this (0 = off)")

    g = p.add_argument_group("features / matching")
    g.add_argument("--max-keypoints", type=int, default=512)
    g.add_argument("--nms-radius", type=int, default=4)
    g.add_argument("--keypoint-threshold", type=float, default=0.005)
    g.add_argument("--sinkhorn-iters", type=int, default=10)
    g.add_argument("--match-threshold", type=float, default=0.2)
    g.add_argument("--superglue-weights", choices=["indoor", "outdoor"], default="outdoor")
    g.add_argument("--superglue-path", default="../SuperGluePretrainedNetwork",
                   help="path to a clone of magicleap/SuperGluePretrainedNetwork")
    g.add_argument("--ratio", type=float, default=0.75, help="ORB Lowe ratio")
    g.add_argument("--ransac-thresh", type=float, default=5.0, help="RANSAC reprojection px")

    p.add_argument("--repeat", type=int, default=3,
                   help="timed matching runs (first run is reported separately as warm-up)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load ----
    t = time.perf_counter()
    drone0 = cv2.imread(args.drone, cv2.IMREAD_GRAYSCALE)
    map0 = cv2.imread(args.map, cv2.IMREAD_GRAYSCALE)
    if drone0 is None or map0 is None:
        sys.exit("Could not read one of the input images.")
    t_load = time.perf_counter() - t

    # ---- resize ----
    t = time.perf_counter()
    drone, s_drone = resize_max_side(drone0, args.drone_max_side)
    mapimg, s_map = resize_max_side(map0, args.map_max_side)
    t_resize = time.perf_counter() - t
    print(f"drone {drone0.shape[::-1]} -> {drone.shape[::-1]} | "
          f"map {map0.shape[::-1]} -> {mapimg.shape[::-1]} (w,h)")

    # ---- model ----
    t = time.perf_counter()
    run = build_superglue(args) if args.method == "superglue" else build_orb(args)
    t_init = time.perf_counter() - t

    # ---- match (timed) ----
    times = []
    for _ in range(max(1, args.repeat)):
        t = time.perf_counter()
        p0, p1, conf = run(drone, mapimg)
        times.append(time.perf_counter() - t)

    # ---- geometry ----
    t = time.perf_counter()
    loc = estimate_location(p0, p1, drone.shape, args.ransac_thresh)
    t_geom = time.perf_counter() - t

    # ---- report ----
    n_inl = int(loc["inlier_mask"].sum()) if loc else 0
    steady = times[1:] if len(times) > 1 else times
    summary = {
        "method": args.method,
        "device": args.device if args.method == "superglue" else "cpu",
        "drone_size_original": list(drone0.shape[::-1]),
        "map_size_original": list(map0.shape[::-1]),
        "drone_size_used": list(drone.shape[::-1]),
        "map_size_used": list(mapimg.shape[::-1]),
        "matches": int(len(p0)),
        "inliers": n_inl,
        "quad_convex": loc["quad_convex"] if loc else None,
        "timing_ms": {
            "load_images": round(t_load * 1e3, 1),
            "resize": round(t_resize * 1e3, 1),
            "model_init": round(t_init * 1e3, 1),
            "match_first_run": round(times[0] * 1e3, 1),
            "match_mean_steady": round(float(np.mean(steady)) * 1e3, 1),
            "homography": round(t_geom * 1e3, 2),
        },
        "settings": {k: v for k, v in vars(args).items()
                     if k in ("max_keypoints", "sinkhorn_iters", "match_threshold",
                              "drone_max_side", "map_max_side", "ratio", "ransac_thresh")},
    }

    if loc:
        # express results in ORIGINAL map pixels
        center_orig = loc["center"] / s_map
        quad_orig = loc["quad"] / s_map
        summary["center_px_original_map"] = [round(float(v), 1) for v in center_orig]
        summary["footprint_px_original_map"] = [[round(float(x), 1), round(float(y), 1)]
                                                for x, y in quad_orig]

        inliers = loc["inlier_mask"]
        cv2.imwrite(str(out / "matches.png"), draw_matches(drone, mapimg, p0, p1, inliers))
        overlay = draw_overlay(mapimg, loc, f"({center_orig[0]:.0f}, {center_orig[1]:.0f})")
        cv2.imwrite(str(out / "overlay.png"), overlay)
        zoom = zoom_crop(overlay, loc["quad"])
        if zoom is not None:
            cv2.imwrite(str(out / "zoom.png"), zoom)
    else:
        print("Localization failed: not enough matches for a homography.")

    (out / "log.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
