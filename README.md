# Drone-to-Map Localization

Locate a small aerial (drone) image inside a large map image using local feature
matching and homography estimation. Two matchers are provided so the
accuracy/speed trade-off can be compared directly:

| Method | Features | Matcher | Character |
|---|---|---|---|
| `superglue` | SuperPoint | SuperGlue | learned, robust to viewpoint/appearance change, slow on CPU |
| `orb` | ORB | BFMatcher + Lowe ratio | classical, fast, needs texture and similar scale |

## Pipeline

```
drone image ─┐
             ├─> downscale ─> keypoints + descriptors ─> matching ─> RANSAC homography ─> footprint + center on map
map image  ──┘
```

1. Load both images in grayscale and downscale them (the map can be 8K, so this dominates run time).
2. Detect keypoints and match them (SuperPoint + SuperGlue, or ORB).
3. Estimate a homography with RANSAC, project the drone image corners onto the map, and check the footprint is a convex quadrilateral.
4. Convert the result back to **original map pixel coordinates**.

## Results

> Fill this table in with numbers from your own run (`results/log.json`).
> Please only report measured values.

| Method | Device | Map used | Keypoints | Sinkhorn iters | Matches | Inliers | Match time (ms) |
|---|---|---|---|---|---|---|---|
| SuperGlue (baseline) | CPU | 4096 x 2160 | 2048 | 20 | TBD | TBD | ~50 000 |
| SuperGlue (tuned) | CPU | TBD | TBD | TBD | TBD | TBD | TBD |
| ORB | CPU | TBD | TBD | - | TBD | TBD | TBD |

![matches](results/matches.png)
![overlay](results/overlay.png)
![zoom](results/zoom.png)

*(Images are written by `localize.py` into `results/`; commit them after running.)*

## Installation

```bash
git clone https://github.com/Bangash514/drone-map-localization.git
cd drone-map-localization
pip install -r requirements.txt

# only for --method superglue
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git ../SuperGluePretrainedNetwork
```

SuperGlue is distributed under its own (non-commercial research) license and is
**not** included in this repository.

## Usage

```bash
# fast classical baseline
python localize.py --drone data/crop_drone.jpg --map data/full_image.jpg \
    --out results --method orb --map-max-side 1500 --max-keypoints 1000

# learned matcher, reduced settings for CPU
python localize.py --drone data/crop_drone.jpg --map data/full_image.jpg \
    --out results --method superglue --superglue-path ../SuperGluePretrainedNetwork \
    --map-max-side 2000 --max-keypoints 512 --sinkhorn-iters 10
```

Outputs in `--out`:

- `matches.png` – side-by-side matches (green = RANSAC inliers, red = outliers)
- `overlay.png` – map with the estimated drone footprint and center
- `zoom.png` – crop around the estimated area
- `log.json` – matches, inliers, timings per stage, settings, and the location in original map pixels

## Speeding it up

The baseline took about 50 s on CPU. The largest levers, in the order I would try them:

1. **Map resolution** (`--map-max-side`): SuperGlue cost grows quickly with the number of keypoints, which grows with image area.
2. **Keypoints** (`--max-keypoints`): fewer keypoints shrink the attention/assignment problem.
3. **Sinkhorn iterations** (`--sinkhorn-iters`): 20 by default in the reference code; fewer is faster, with some loss in match quality.
4. **GPU** (`--device cuda`) and, beyond that, ONNX/TensorRT export.
5. **Coarse-to-fine search** for large maps: match on a thumbnail first, then re-match only the candidate region at full resolution.

`log.json` reports `match_first_run` separately from `match_mean_steady`, because
the first call includes one-time initialization. Real-time budgets (e.g. <100 ms)
are only realistic with a GPU or a lighter matcher; check the numbers on your own hardware.

## Limitations

- Downscaling the map loses detail. If the drone image covers a small area, there may be too few matches at low resolution.
- A single homography assumes a roughly planar scene and a near-nadir view.
- Large changes in season, lighting or altitude reduce the match count; the ORB path is more sensitive to this than SuperGlue.

## Notes

I also reviewed [TESSERA](https://github.com/ucam-eo/tessera) while working on this. It is
a geospatial foundation model that produces embeddings from Sentinel-1/2 time series,
so it targets a different kind of task (land-cover-style analysis) than pairwise image
matching and is not used in this pipeline.

## Author

Owais Muhammad (Bangash514) – software / algorithm engineering for medical imaging and robotics.
