# Date: 2026-11-02
# Change aim: scale-aware drone resize, offline ORB map index, fast index query
# File: localize.py

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ORB_KW = dict(scaleFactor=1.2, nlevels=8, edgeThreshold=31, patchSize=31, fastThreshold=20)
TILE = 1024
MARGIN = 128
MIN_INLIERS = 8
DEFAULT_SCALES = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0)


def ms_since(t):
    return (time.perf_counter() - t) * 1e3


def resize_max_side(img, max_side):
    h, w = img.shape[:2]
    scale = 1.0 if not max_side else min(1.0, max_side / float(max(h, w)))
    if scale == 1.0:
        return img, 1.0
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), scale


def resize_scale(img, scale):
    if abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, size, interpolation=interp)


def estimate_location(p0, p1, drone_shape, thresh):
    if len(p0) < 4:
        return None
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, thresh)
    if H is None or mask is None:
        return None
    h, w = drone_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    center = cv2.perspectiveTransform(np.float32([[[w / 2.0, h / 2.0]]]), H).reshape(2)
    convex = bool(cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)))
    inl = mask.ravel().astype(bool)
    return {"H": H, "inlier_mask": inl, "inliers": int(inl.sum()), "quad": quad,
            "center": center, "quad_convex": convex,
            "success": bool(convex and inl.sum() >= MIN_INLIERS)}


def detect_tiled(gray, per_tile):
    h, w = gray.shape
    orb = cv2.ORB_create(nfeatures=per_tile, **ORB_KW)
    xy, des = [], []
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            x1, y1 = min(x0 + TILE, w), min(y0 + TILE, h)
            px0, py0 = max(0, x0 - MARGIN), max(0, y0 - MARGIN)
            px1, py1 = min(w, x1 + MARGIN), min(h, y1 + MARGIN)
            kps, d = orb.detectAndCompute(np.ascontiguousarray(gray[py0:py1, px0:px1]), None)
            if d is None or not kps:
                continue
            pts = np.float32([k.pt for k in kps]) + np.float32([px0, py0])
            keep = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
            xy.append(pts[keep])
            des.append(d[keep])
    return np.vstack(xy), np.vstack(des)


def write_idx(path, xy, des, size, build_ms):
    with open(path, "wb") as f:
        f.write(b"DVLIDX01")
        f.write(np.array([size[0], size[1], len(xy)], np.int32).tobytes())
        f.write(np.array([build_ms], np.float64).tobytes())
        f.write(np.ascontiguousarray(xy, np.float32).tobytes())
        f.write(np.ascontiguousarray(des, np.uint8).tobytes())


def build_index(map_path, out_path, per_tile=800, export_idx=None):
    t0 = time.perf_counter()
    gray = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        sys.exit(f"cannot read map: {map_path}")
    t_read = ms_since(t0)
    xy, des = detect_tiled(gray, per_tile)
    build_ms = round(ms_since(t0), 1)
    np.savez(out_path, xy=xy, des=des, size=np.array([gray.shape[1], gray.shape[0]]),
             params=np.array([TILE, MARGIN, per_tile]), build_ms=np.array([build_ms]))
    if export_idx:
        write_idx(export_idx, xy, des, (gray.shape[1], gray.shape[0]), build_ms)
    return {"index_build_ms": build_ms, "map_read_ms": round(t_read, 1),
            "map_size": [int(gray.shape[1]), int(gray.shape[0])],
            "n_keypoints": int(len(xy)), "tile": TILE, "margin": MARGIN,
            "features_per_tile": per_tile, "index_file": str(out_path)}


class MapIndex:
    def __init__(self, path, matcher="bf"):
        t0 = time.perf_counter()
        z = np.load(path)
        self.xy = z["xy"]
        self.des = np.ascontiguousarray(z["des"])
        self.size = [int(v) for v in z["size"]]
        self.build_ms = float(z["build_ms"][0]) if "build_ms" in z.files else None
        self.kind = matcher
        if matcher == "flann":
            self.matcher = cv2.FlannBasedMatcher(
                dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
                dict(checks=50))
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.matcher.add([self.des])
        self.matcher.train()
        self.load_ms = ms_since(t0)

    def query(self, drone_gray, nfeatures=1500, ratio=0.75, ransac=5.0):
        t = time.perf_counter()
        orb = cv2.ORB_create(nfeatures=nfeatures, **ORB_KW)
        kps, des = orb.detectAndCompute(drone_gray, None)
        detect_ms = ms_since(t)
        if des is None or len(kps) < 4:
            return {"loc": None, "n_drone_kp": 0, "matches": 0, "detect_ms": detect_ms,
                    "match_ms": 0.0, "homography_ms": 0.0}
        t = time.perf_counter()
        pairs = self.matcher.knnMatch(des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
        match_ms = ms_since(t)
        t = time.perf_counter()
        loc = None
        if len(good) >= 4:
            p0 = np.float32([kps[m.queryIdx].pt for m in good])
            p1 = self.xy[[m.trainIdx for m in good]]
            loc = estimate_location(p0, p1, drone_gray.shape, ransac)
        hom_ms = ms_since(t)
        return {"loc": loc, "n_drone_kp": len(kps), "matches": len(good),
                "detect_ms": detect_ms, "match_ms": match_ms, "homography_ms": hom_ms}

    def localize(self, drone_gray, scales=(1.0,), **kw):
        best = None
        tot = [0.0, 0.0, 0.0]
        for sc in scales:
            d = resize_scale(drone_gray, sc)
            if min(d.shape[:2]) < 32:
                continue
            r = self.query(d, **kw)
            r["scale"] = sc
            tot[0] += r["detect_ms"]
            tot[1] += r["match_ms"]
            tot[2] += r["homography_ms"]
            inl = r["loc"]["inliers"] if r["loc"] else 0
            if best is None or inl > best[0]:
                best = (inl, r)
        if best is None:
            return None
        out = best[1]
        out["detect_ms"], out["match_ms"], out["homography_ms"] = tot
        out["scales_tried"] = len(scales)
        return out


def query_summary(res, index, drone_shape, load_drone_ms):
    loc = res["loc"] if res else None
    d = res or {"detect_ms": 0.0, "match_ms": 0.0, "homography_ms": 0.0, "matches": 0,
                "n_drone_kp": 0, "scale": None}
    total = d["detect_ms"] + d["match_ms"] + d["homography_ms"]
    out = {
        "method": "orb-index", "matcher": index.kind,
        "map_size": index.size, "drone_size": [int(drone_shape[1]), int(drone_shape[0])],
        "index_keypoints": int(len(index.xy)), "drone_keypoints": d["n_drone_kp"],
        "drone_scale_applied": d["scale"], "matches": d["matches"],
        "inliers": loc["inliers"] if loc else 0,
        "quad_convex": loc["quad_convex"] if loc else None,
        "success": loc["success"] if loc else False,
        "timing_ms": {
            "index_build_ms": index.build_ms,
            "index_load_ms": round(index.load_ms, 1),
            "drone_load_ms": round(load_drone_ms, 1),
            "query_detect_ms": round(d["detect_ms"], 1),
            "query_match_ms": round(d["match_ms"], 1),
            "homography_ms": round(d["homography_ms"], 2),
            "total_query_ms": round(total, 1),
        },
        "total_query_under_100ms": bool(total < 100.0),
    }
    if loc:
        s = d["scale"]
        out["center_px_map"] = [round(float(v), 2) for v in loc["center"]]
        out["footprint_px_map"] = [[round(float(x), 1), round(float(y), 1)] for x, y in loc["quad"]]
        out["drone_scale_applied"] = s
    return out


def build_superglue(args):
    import torch
    sg_path = Path(args.superglue_path).expanduser().resolve()
    if not (sg_path / "models" / "matching.py").exists():
        sys.exit(f"SuperGlue repo not found at {sg_path}")
    sys.path.insert(0, str(sg_path))
    from models.matching import Matching
    torch.set_grad_enabled(False)
    config = {
        "superpoint": {"nms_radius": args.nms_radius, "keypoint_threshold": args.keypoint_threshold,
                       "max_keypoints": args.max_keypoints},
        "superglue": {"weights": args.superglue_weights, "sinkhorn_iterations": args.sinkhorn_iters,
                      "match_threshold": args.match_threshold},
    }
    model = Matching(config).eval().to(args.device)

    def to_tensor(g):
        return torch.from_numpy(g.astype(np.float32) / 255.0)[None, None].to(args.device)

    def run(g0, g1):
        pred = model({"image0": to_tensor(g0), "image1": to_tensor(g1)})
        k0 = pred["keypoints0"][0].cpu().numpy()
        k1 = pred["keypoints1"][0].cpu().numpy()
        m = pred["matches0"][0].cpu().numpy()
        c = pred["matching_scores0"][0].cpu().numpy()
        v = m > -1
        return k0[v], k1[m[v]], c[v]

    return run


def build_orb_pair(args):
    orb_d = cv2.ORB_create(nfeatures=args.max_keypoints, **ORB_KW)
    orb_m = cv2.ORB_create(nfeatures=args.map_keypoints, **ORB_KW)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def run(g0, g1):
        k0, d0 = orb_d.detectAndCompute(g0, None)
        k1, d1 = orb_m.detectAndCompute(g1, None)
        empty = (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32))
        if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2:
            return empty
        good = [p[0] for p in bf.knnMatch(d0, d1, k=2)
                if len(p) == 2 and p[0].distance < args.ratio * p[1].distance]
        if not good:
            return empty
        p0 = np.float32([k0[m.queryIdx].pt for m in good])
        p1 = np.float32([k1[m.trainIdx].pt for m in good])
        return p0, p1, np.float32([1.0 - m.distance / 256.0 for m in good])

    return run


def draw_matches(g0, g1, p0, p1, inliers, max_lines=150, height=700):
    s0, s1 = height / g0.shape[0], height / g1.shape[0]
    a = cv2.cvtColor(cv2.resize(g0, None, fx=s0, fy=s0), cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(cv2.resize(g1, None, fx=s1, fy=s1), cv2.COLOR_GRAY2BGR)
    canvas = cv2.hconcat([a, b])
    for i in np.argsort(~inliers)[:max_lines]:
        x0, y0 = p0[i] * s0
        x1, y1 = p1[i] * s1
        color = (0, 220, 0) if inliers[i] else (0, 0, 220)
        cv2.line(canvas, (int(x0), int(y0)), (int(x1 + a.shape[1]), int(y1)), color, 1, cv2.LINE_AA)
    return canvas


def draw_overlay(map_gray, quad, center, label):
    overlay = cv2.cvtColor(map_gray, cv2.COLOR_GRAY2BGR)
    th = max(2, overlay.shape[1] // 600)
    cv2.polylines(overlay, [quad.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), th, cv2.LINE_AA)
    cx, cy = center.astype(int)
    cv2.circle(overlay, (cx, cy), th * 3, (0, 0, 255), -1)
    cv2.putText(overlay, label, (cx + 12, cy - 12), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, overlay.shape[1] / 2500), (0, 255, 0), th, cv2.LINE_AA)
    return overlay


def zoom_crop(overlay, quad, margin=0.5):
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    H, W = overlay.shape[:2]
    x0, x1 = int(max(0, x0 - mx)), int(min(W, x1 + mx))
    y0, y1 = int(max(0, y0 - my)), int(min(H, y1 + my))
    return overlay[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else None


def cmd_build_index(args):
    stats = build_index(args.map, args.out, args.features_per_tile, args.export_idx)
    print(json.dumps(stats, indent=2))


def cmd_query(args):
    t = time.perf_counter()
    drone = cv2.imread(args.drone, cv2.IMREAD_GRAYSCALE)
    if drone is None:
        sys.exit(f"cannot read drone image: {args.drone}")
    load_drone_ms = ms_since(t)
    index = MapIndex(args.index, args.matcher)
    if args.drone_scale_rel:
        scales = (args.drone_scale_rel,)
    elif args.scale_search:
        scales = tuple(args.scale_candidates)
    else:
        scales = (1.0,)
    runs = []
    for _ in range(max(1, args.repeat)):
        runs.append(index.localize(drone, scales, nfeatures=args.features, ratio=args.ratio,
                                   ransac=args.ransac_thresh))
    summaries = [query_summary(r, index, drone.shape, load_drone_ms) for r in runs]
    out = summaries[-1]
    if len(summaries) > 1:
        for k in out["timing_ms"]:
            vals = [s["timing_ms"][k] for s in summaries if s["timing_ms"][k] is not None]
            if vals and k not in ("index_load_ms", "drone_load_ms", "index_build_ms"):
                out["timing_ms"][k] = round(statistics.median(vals), 2)
        out["timing_repeat"] = len(summaries)
        out["total_query_under_100ms"] = bool(out["timing_ms"]["total_query_ms"] < 100.0)
    out["settings"] = {"features": args.features, "ratio": args.ratio,
                       "ransac_thresh": args.ransac_thresh, "scales": list(scales)}
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.map and runs[-1] and runs[-1]["loc"]:
        m0 = cv2.imread(args.map, cv2.IMREAD_GRAYSCALE)
        if m0 is not None:
            m, s = resize_max_side(m0, 2000)
            loc = runs[-1]["loc"]
            c = loc["center"]
            ov = draw_overlay(m, loc["quad"] * s, c * s, f"({c[0]:.0f}, {c[1]:.0f})")
            cv2.imwrite(str(outdir / "overlay.png"), ov)
    (outdir / "log.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def cmd_match(args):
    t0 = time.perf_counter()
    drone0 = cv2.imread(args.drone, cv2.IMREAD_GRAYSCALE)
    map0 = cv2.imread(args.map, cv2.IMREAD_GRAYSCALE)
    if drone0 is None or map0 is None:
        sys.exit("cannot read an input image")
    load_ms = ms_since(t0)
    t = time.perf_counter()
    mapimg, s_map = resize_max_side(map0, args.map_max_side)
    resize_ms = ms_since(t)
    t = time.perf_counter()
    run = build_superglue(args) if args.method == "superglue" else build_orb_pair(args)
    init_ms = ms_since(t)
    rels = [args.drone_scale_rel] if args.drone_scale_rel else list(args.scale_candidates)
    table, best = [], None
    for rel in rels:
        applied = rel * s_map
        d = resize_scale(drone0, applied)
        if min(d.shape[:2]) < 32:
            table.append({"drone_scale_rel": rel, "skipped": "drone smaller than 32 px"})
            continue
        t = time.perf_counter()
        p0, p1, _ = run(d, mapimg)
        match_ms = ms_since(t)
        t = time.perf_counter()
        loc = estimate_location(p0, p1, d.shape, args.ransac_thresh)
        hom_ms = ms_since(t)
        inl = loc["inliers"] if loc else 0
        table.append({"drone_scale_rel": rel, "drone_scale_applied": round(applied, 5),
                      "drone_size_used": [int(d.shape[1]), int(d.shape[0])],
                      "matches": int(len(p0)), "inliers": inl,
                      "quad_convex": loc["quad_convex"] if loc else None,
                      "match_ms": round(match_ms, 1), "homography_ms": round(hom_ms, 2)})
        if best is None or inl > best["inl"]:
            best = {"inl": inl, "rel": rel, "d": d, "p0": p0, "p1": p1, "loc": loc, "applied": applied}
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": args.method, "map_size_original": [int(map0.shape[1]), int(map0.shape[0])],
        "map_size_used": [int(mapimg.shape[1]), int(mapimg.shape[0])], "map_scale": round(s_map, 5),
        "drone_size_original": [int(drone0.shape[1]), int(drone0.shape[0])],
        "scale_search": table,
        "timing_ms": {"load_images": round(load_ms, 1), "resize_map": round(resize_ms, 1),
                      "model_init": round(init_ms, 1),
                      "match_all_scales": round(sum(r.get("match_ms", 0) for r in table), 1)},
    }
    if best and best["loc"]:
        loc = best["loc"]
        center = loc["center"] / s_map
        summary.update({
            "drone_scale_rel_chosen": best["rel"], "drone_scale_applied": round(best["applied"], 5),
            "inliers": loc["inliers"], "quad_convex": loc["quad_convex"], "success": loc["success"],
            "center_px_original_map": [round(float(v), 2) for v in center],
            "footprint_px_original_map": [[round(float(x), 1), round(float(y), 1)]
                                          for x, y in loc["quad"] / s_map]})
        cv2.imwrite(str(outdir / "matches.png"),
                    draw_matches(best["d"], mapimg, best["p0"], best["p1"], loc["inlier_mask"]))
        ov = draw_overlay(mapimg, loc["quad"], loc["center"], f"({center[0]:.0f}, {center[1]:.0f})")
        cv2.imwrite(str(outdir / "overlay.png"), ov)
        z = zoom_crop(ov, loc["quad"])
        if z is not None:
            cv2.imwrite(str(outdir / "zoom.png"), z)
    else:
        summary.update({"success": False, "inliers": 0})
    (outdir / "log.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Locate a drone image inside a large map image.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="offline: ORB keypoints of the whole map -> .npz")
    b.add_argument("--map", required=True)
    b.add_argument("--out", default="index.npz")
    b.add_argument("--features-per-tile", type=int, default=800)
    b.add_argument("--export-idx", help="also write the binary .idx format read by the C++ tool")
    b.set_defaults(fn=cmd_build_index)

    q = sub.add_parser("query", help="online: match a drone image against a prebuilt index")
    q.add_argument("--index", required=True)
    q.add_argument("--drone", required=True)
    q.add_argument("--out", default="results")
    q.add_argument("--map", help="optional, only to draw overlay.png")
    q.add_argument("--matcher", choices=["bf", "flann"], default="bf")
    q.add_argument("--features", type=int, default=1500)
    q.add_argument("--ratio", type=float, default=0.75)
    q.add_argument("--ransac-thresh", type=float, default=5.0)
    q.add_argument("--drone-scale-rel", type=float, help="map px per drone px; drone is resized by this")
    q.add_argument("--scale-search", action="store_true", help="try --scale-candidates, keep most inliers")
    q.add_argument("--scale-candidates", type=float, nargs="+", default=list(DEFAULT_SCALES))
    q.add_argument("--repeat", type=int, default=1, help="repeat query, report median timings")
    q.set_defaults(fn=cmd_query)

    m = sub.add_parser("match", help="direct pair matching (orb or superglue), no index")
    m.add_argument("--drone", required=True)
    m.add_argument("--map", required=True)
    m.add_argument("--out", default="results")
    m.add_argument("--method", choices=["orb", "superglue"], default="orb")
    m.add_argument("--device", default="cpu")
    m.add_argument("--map-max-side", type=int, default=2000)
    m.add_argument("--drone-scale-rel", type=float, help="map px per drone px at full resolution")
    m.add_argument("--scale-candidates", type=float, nargs="+", default=list(DEFAULT_SCALES))
    m.add_argument("--max-keypoints", type=int, default=1000, help="keypoints on the drone image")
    m.add_argument("--map-keypoints", type=int, default=20000, help="ORB keypoints on the map image")
    m.add_argument("--nms-radius", type=int, default=4)
    m.add_argument("--keypoint-threshold", type=float, default=0.005)
    m.add_argument("--sinkhorn-iters", type=int, default=10)
    m.add_argument("--match-threshold", type=float, default=0.2)
    m.add_argument("--superglue-weights", choices=["indoor", "outdoor"], default="outdoor")
    m.add_argument("--superglue-path", default="../SuperGluePretrainedNetwork")
    m.add_argument("--ratio", type=float, default=0.75)
    m.add_argument("--ransac-thresh", type=float, default=5.0)
    m.set_defaults(fn=cmd_match)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    a.fn(a)
