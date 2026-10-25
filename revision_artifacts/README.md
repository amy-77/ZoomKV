# VLDB revision artifacts

Raw measurements referenced in the ZoomKV revision response letter.

| File | Used in | Contents |
|---|---|---|
| `cost_overhead_calibration_kernel_h20.json` | Reviewer 6, D7; Section 4.7 | Per-level chunk-selection kernel time on an NVIDIA H20 for parent and child levels with 128–8192 scored chunks. Each row is `[scored_chunks, selected_chunks, mean_us, min_us, max_us]` (top-k keeps 30% of child chunks and 40% of parent chunks), plus the linear fit `t = intercept + slope * chunks`. The intercept divided by the slope (about 2,500 chunks) is the per-level fixed overhead `C0`. |
| `r6_default_throughput_128k.json` | Reviewer 7 (R6 throughput) | TTFT, TPOT and decode throughput of ZoomKV, ParisKV, FreeKV, ShadowKV and RetroInfer on one A100 80GB with Qwen3-4B-Thinking-2507, 131,072 input tokens, batch size 1, 32 warmup and 128 timed decode tokens, 3 repetitions. |
