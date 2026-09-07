**Caveat -- the number below is not a measurement of the reasoning faculty.** There are
two independent reasons for that, and a reader who takes nothing else from this card
should take these.

*This region has not been trained for reasoning.* It is at present trained and scored as
a bi-encoder retriever, exactly like `language`, `compress` and `retrieve`: the harness
has a single contrastive training path and this region takes that generic path, and the
promotion metric below, `held_out.recall@1`, is a retrieval metric. **No
reasoning-specific training objective and no reasoning-specific evaluation battery exist
in this project yet.** What the figure describes is a weak retriever of worked
derivations -- a TF-IDF bag-of-words baseline reaches 0.8730 on the same battery.

*A standalone bar is not a meaningful gate for this region at all.* It is designed to
operate on workspace latents produced by the composed mind, a distribution that does not
exist until the composite does. Training it alone therefore feeds it a distribution it
never sees in production, and scoring it alone measures it outside its design
conditions. The figure below is the artifact of pre-training something that cannot yet
be meaningfully pre-trained and then scoring it on a retrieval metric.

So `0.1270` is correctly computed, and it is **not evidence that the reasoning faculty
is weak** -- that faculty has not been trained at all yet. It is equally not evidence
that nothing is wrong: a genuine training defect would still have to be fixed on its own
terms. Read it as one thing only -- a dated, annotated baseline, to be re-run through the
composite and compared against a reasoning-specific objective and a step-sensitive
battery once those exist.

main_rule best_primary_metric_all_gates_pass on train:$.held_out.recall@1, applied within the same code pin (7bc2699), the same corpus fingerprint (ca364a92d) and the production batch for this region (512; batch 1280 ran out of memory at max_len 256), among the reason cells whose train, test, quantize and test-quant stages all passed. The seed axis resamples the held-out split (untrained baselines 0.0059 at seed 0 vs 0.0078 at seed 1); seed 0 is the canonical split and is promoted: reason-b512-s0 (0.1270; seed 1 scored 0.1426 on its own split). The batch-256 cells scored higher on their splits (0.1875 at seed 0, 0.1523 at seed 1) and are excluded from this promotion because the production batch is 512; that gap is recorded as an open finding about the region, not hidden. This region is the weakest in the matrix by a wide margin and is under a separate improvement study. Receipts are csd-metrics/v1. Promoted variant: reason-b512-s0-7bc2699-20260904-cca364a92.
