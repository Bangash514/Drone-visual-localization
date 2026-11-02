# Date: 2026-11-02
# Change aim: synthetic map and drone-crop generator with known ground truth
# File: tests/simlib.py

import math

import cv2
import numpy as np


def make_map(w, h, seed=0):
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for octave in range(2, 9):
        gh, gw = max(2, h >> octave), max(2, w >> octave)
        layer = rng.random((gh, gw), dtype=np.float32)
        acc += amp * cv2.resize(layer, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.7
    img = (acc / total)
    img = cv2.normalize(img, None, 40, 200, cv2.NORM_MINMAX).astype(np.uint8)
    n_shapes = int(w * h / 4000)
    for _ in range(n_shapes):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        c = int(rng.integers(10, 245))
        kind = int(rng.integers(0, 4))
        if kind == 0:
            dx, dy = int(rng.integers(6, 60)), int(rng.integers(6, 60))
            cv2.rectangle(img, (x, y), (x + dx, y + dy), c, -1)
        elif kind == 1:
            r = int(rng.integers(4, 30))
            cv2.circle(img, (x, y), r, c, -1)
        elif kind == 2:
            pts = np.array([[x + int(rng.integers(-40, 40)), y + int(rng.integers(-40, 40))]
                            for _ in range(4)], np.int32)
            cv2.fillConvexPoly(img, cv2.convexHull(pts), c)
        else:
            ang = rng.uniform(0, math.pi)
            ln = int(rng.integers(40, 300))
            cv2.line(img, (x, y), (int(x + ln * math.cos(ang)), int(y + ln * math.sin(ang))), c,
                     int(rng.integers(2, 7)))
    noise = rng.normal(0, 3, (h, w)).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def make_drone(world, cx, cy, scale, angle, alpha=1.0, beta=0.0, size=(800, 500), noise=2.0, seed=0):
    dw, dh = size
    r = int(math.ceil(0.5 * math.hypot(dw, dh) / scale)) + 8
    x0, y0 = int(cx) - r, int(cy) - r
    x1, y1 = int(cx) + r, int(cy) + r
    if x0 < 0 or y0 < 0 or x1 > world.shape[1] or y1 > world.shape[0]:
        raise ValueError("crop leaves the map")
    patch = world[y0:y1, x0:x1]
    if scale < 1.0:
        patch = cv2.GaussianBlur(patch, (0, 0), 0.5 / scale)
    lc = (cx - x0, cy - y0)
    M = cv2.getRotationMatrix2D(lc, angle, scale)
    M[0, 2] += dw / 2.0 - lc[0]
    M[1, 2] += dh / 2.0 - lc[1]
    out = cv2.warpAffine(patch, M, (dw, dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rng = np.random.default_rng(seed)
    f = out.astype(np.float32) * alpha + beta + rng.normal(0, noise, out.shape).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


def make_cases(w, h, n, seed=1, size=(800, 500)):
    rng = np.random.default_rng(seed)
    scales = [0.5, 0.65, 0.8, 1.0]
    cases = []
    i = 0
    while len(cases) < n:
        s = scales[i % len(scales)]
        i += 1
        r = int(math.ceil(0.5 * math.hypot(*size) / s)) + 16
        if w < 2 * r + 2 or h < 2 * r + 2:
            continue
        cx = float(rng.uniform(r, w - r))
        cy = float(rng.uniform(r, h - r))
        cases.append({
            "id": len(cases), "cx": round(cx, 1), "cy": round(cy, 1), "scale": s,
            "angle": round(float(rng.uniform(-10, 10)), 2),
            "alpha": round(float(rng.uniform(0.8, 1.2)), 3),
            "beta": round(float(rng.uniform(-20, 20)), 1),
            "seed": int(rng.integers(0, 1 << 30)),
        })
    return cases
