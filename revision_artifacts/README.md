# VLDB revision artifacts

Raw measurements referenced in the ZoomKV revision response letter.

| File | Used in | Contents |
|---|---|---|
| `cost_overhead_calibration_kernel_h20.json` | Reviewer 6, D7; Section 4.7 | Per-level chunk-selection kernel time on an NVIDIA H20 for parent and child levels with 128–8192 scored chunks. Each row is `[scored_chunks, selected_chunks, mean_us, min_us, max_us]` (top-k keeps 30% of child chunks and 40% of parent chunks), plus the linear fit `t = intercept + slope * chunks`. The intercept divided by the slope (about 2,500 chunks) is the per-level fixed overhead `C0`. Produced by `calibrate_level_overhead.py` (`quest_chunk_score` + `float_topk_3d`, batch 1, 8 KV heads, head dim 128, bf16; 5 blocks of 200 timed launches per size). |
| `calibrate_level_overhead.py` | Reviewer 6, D7; Section 4.7 | The timing script: `PYTHONPATH=<ZoomKV root> CUDA_VISIBLE_DEVICES=<gpu> python calibrate_level_overhead.py --out out.json`. It times one level's chunk scoring plus chunk top-k for 128–8192 chunks, fits `t = intercept + slope * chunks` per level and reports `intercept / slope` as `C0` in chunks. Run it on an otherwise idle GPU. |
| `cost_overhead_calibration_kernel_a100_rerun_20261029.json` | Reviewer 6, D7 (cross-check) | The same script on an idle A100-40GB (2026-10-29): same curve shape; the per-chunk slope is about 2.2x the H20 value, in line with the bandwidth ratio of the two GPUs, so `C0` in chunk units is device dependent (about 1,200 chunks on A100, about 2,500 on H20). The cost model only uses `C0` as a common offset across two-level candidates, so this does not change any selection. |
| `r6_default_throughput_128k.json` | Reviewer 7 (R6 throughput) | TTFT, TPOT and decode throughput of ZoomKV, ParisKV, FreeKV, ShadowKV and RetroInfer on one A100 80GB with Qwen3-4B-Thinking-2507, 131,072 input tokens, batch size 1, 32 warmup and 128 timed decode tokens, 3 repetitions. |
