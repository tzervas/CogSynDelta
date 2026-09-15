Promoted variant: visual-b128-s1-3ce18db-20260905-cab5761b6. It was chosen by main_rule best_primary_metric_all_gates_pass on train:$.held_out.top1 among the visual cells of this run whose train, test, quantize and test-quant stages all passed. Both candidates pass PREREG g22 H1, and the gap between them is inside seed-to-seed spread.

Candidate pool: run 2 (pin 3ce18db, corpus visual-clean-v1 ab5761b65714e4ba4d7c36df095f3599, seeds 0 and 1, steps 4000, batch 128).

| seed | train:$.held_out.top1 | selected |
|---|---|---|
| 1 | 0.7237 | yes |
| 0 | 0.7224 | no |

The table ranks the eligible cells by the primary metric.

The run-1 cells (pin 198074a) are excluded: they have no eval or quant receipt and no checkpoint binding (see ../g22-visual-prereg-run1-2026-09-05/).
