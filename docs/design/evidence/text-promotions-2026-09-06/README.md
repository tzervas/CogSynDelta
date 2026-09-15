# Text-region promotions, batch 1: proposal (nothing promoted yet)

## Summary

Four text regions have a candidate for `region_main`. This directory holds, per region, the
selection text the promoted card would print, and a preview of that card rendered with the
lexical baseline column. The operator decides two things before anything is promoted:

1. The comparability rule. The drafts apply: same code pin and corpus fingerprint, the
   production batch, and seed 0 as the canonical split. Under the alternative "maximum across
   seeds", language and reason would flip to seed 1.
2. Whether to promote at all, given that three of the four sit at the bag-of-words ceiling of
   their own battery (the card now says so).

## Candidates

| region | cell | model recall@1 | untrained | TF-IDF | model over ceiling |
|---|---|---|---|---|---|
| language | code-b1280-s0-7bc2699-20260904 | 0.986 | 0.229 | 0.982 | 1.00 |
| compress | compress-b1280-s0-7bc2699-20260904 | 0.777 | 0.037 | 0.713 | 1.09 |
| retrieve | retrieve-b1280-s0-7bc2699-20260904 | 0.756 | 0.002 | 0.770 | 0.98 |
| reason | reason-b512-s0-7bc2699-20260904 | 0.127 | 0.006 | 0.873 | 0.15 |

Numbers are the fp32 eval receipts and the lexical sidecar receipts written on 2026-09-06.

## Files

- `HOW-CHOSEN-<region>.md`: the selection text (the `--how-chosen-file` input).
- `PREVIEW-<region>-region_main.md`: the card as it would be published, rendered with tag
  `v0.1.0`.
- `PICKS.txt`: region, cell, variant id, Hub repo, text file (the promotion script's input).

## Known constraint

The matrix's `run.code.sha` now points at the visual pin, so these cells sit under the
previous pin. The harness names them as "not addressed by this config"; promoting them needs
either the promote command to address cells by variant id regardless of the pin (to be
verified) or a temporary pin. That is checked before any promotion runs.
