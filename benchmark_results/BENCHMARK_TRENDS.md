# Benchmark Trend Report

**Generated**: 2026-01-20T02:48:32.101096+00:00
**History Depth**: 5 runs

## Summary

| Status | Count |
|--------|-------|
| 📈 Improving | 2 |
| 📉 Degrading | 0 |
| ➡️ Stable | 14 |

## Metrics by Component

### vl-jepa

| Metric | Trend | Current | Change | Sparkline |
|--------|-------|---------|--------|-----------|
| latency_ms.memory_read | ➡️ | 0.15 | +1.0% | `▁█▆▅▅` |
| latency_ms.memory_write | 📈 | 0.04 | +3.2% | `▁█▁▂▃` |
| latency_ms.vision_encoder_bs1 | ➡️ | 1.23 | +1.6% | `▁▂▄█▄` |
| latency_ms.vision_encoder_bs16 | ➡️ | 5.98 | +4.0% | `▁▃▃█▆` |
| latency_ms.vision_encoder_bs4 | ➡️ | 2.10 | +1.8% | `▁▂▆█▆` |
| latency_ms.vision_encoder_bs8 | ➡️ | 3.43 | +2.4% | `▁█▂▅▃` |
| memory_mb.model_size | ➡️ | 75.04 | +0.0% | `▅▅▅▅▅` |
| memory_mb.peak_allocated | 📈 | 129.60 | +2.2% | `▁▁▁▃█` |
| normalized.param_count | ➡️ | 19671040.00 | +0.0% | `▅▅▅▅▅` |
| normalized.throughput_per_billion_params | ➡️ | 116977.33 | -3.5% | `█▁▃▁▃` |
| throughput.images_per_sec_bs1 | ➡️ | 809.95 | -1.5% | `█▇▆▁▄` |
| throughput.images_per_sec_bs16 | ➡️ | 2657.97 | -4.4% | `█▁▄▁▂` |
| throughput.images_per_sec_bs4 | ➡️ | 1906.06 | -1.4% | `█▇▃▁▁` |
| throughput.images_per_sec_bs8 | ➡️ | 2301.07 | -3.5% | `█▁▃▁▃` |

## Recommendations

- ✅ Overall positive trend: 2 metrics improving
- ➡️ System is stable - good time to establish new baselines
