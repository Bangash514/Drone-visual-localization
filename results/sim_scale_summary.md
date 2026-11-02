| matcher | mode | features | per_tile | index_keypoints | success_rate_pct | mean_error_px | max_error_px | median_detect_ms | median_match_ms | median_total_query_ms | max_total_query_ms | cases_under_100ms_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flann | none | 1500 | 800 | 23938 | 100.0 | 0.55 | 1.28 | 11.4 | 82.9 | 94.9 | 168.9 | 62.5 |
| flann | known | 1500 | 800 | 23938 | 100.0 | 0.4 | 0.91 | 16.7 | 96.7 | 117.8 | 169.6 | 29.2 |
| flann | search | 1500 | 800 | 23938 | 100.0 | 0.39 | 0.91 | 64.8 | 422.0 | 490.4 | 635.6 | 0.0 |
