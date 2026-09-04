# W2c: untrained baselines (2026-09-03)

What this directory records: the two untrained-encoder measurements W2c was scoped to
settle — (i) whether the `code` region's lexical floor is the surviving receipt figure
0.2285 or the unsupported prose figure ≈0.40, and (ii) a real untrained baseline for
`retrieve`'s 0.7480 held-out figure (previously recorded as exactly 0.0000).

`measure_w2c.py` re-instantiates `TextEncoder(vocab 50257, dim 256, depth 4, heads 4)`
against each region's own held-out bin (reproduced from that bin's 2026-09-03 pretrain
receipt — fingerprint, cap-sampling, source counts and dedup all verified to match) and
evaluates recall@1 at two seeds: seed 0 (shared across all four text regions), and a
region-specific seed derived as `int(sha256('csd-w2c-untrained:' + region)[:8 hex], 16)`.
Only the model's `torch.manual_seed` at construction varies between the two; `cfg.seed`
is held at 0 throughout so the held-out bin itself stays fixed. Full detail, including
per-region holdout-reproduction checks and the 14 prose sites carrying the 0.40 claim,
is in `results.json`; `meta.verified` there lists every check this run performed.

## Results (recall@1)

| region   | seed-0 | region-specific seed | chance (1/512) |
|----------|--------|-----------------------|-----------------|
| code     | 0.2285 | 0.2344                | 0.001953        |
| compress | 0.0371 | 0.0410                | 0.001953        |
| retrieve | 0.0020 | 0.0020                | 0.001953        |
| reason   | 0.0059 | 0.0039                | 0.001953        |

`retrieve`'s untrained recall@1 (seed-0 and region-seed alike) lands at exactly 1/512 —
chance on this 512-row diagonal, not the 0.0000 previously on record. That 0.0000 was
one receipt's real measurement, not a bug (an untrained mean-pooled encoder can land at
or below chance on a small held-out diagonal); it is superseded by the 2026-09-03
receipt (`retrieve-20260903T121603Z.json`, recall@1 0.001953125) and by both
measurements here.

τ_lo, chance-normalised (`(s - chance) / (1 - chance)`, evaluated at the
region-specific seed per DEC-36 — see `results.json` → `tau_lo_derivation` for why the
region-specific seed rather than seed 0):

| region   | τ_lo (chance-normalised) |
|----------|---------------------------|
| code     | 0.2329                    |
| compress | 0.0391                    |
| retrieve | 0.0000                    |
| reason   | 0.0020                    |

`retrieve`'s τ_lo rounds to exactly 0 (its region-specific-seed recall@1 equals chance
to the sampled precision), which makes NSRS admission condition (1) barely satisfiable
on that bin — effectively any measured recall@1 above chance clears it. This is a
property of the bin's untrained baseline sitting at chance, not a defect in the
derivation.

## Verdict on the ≈0.40 code lexical floor

**Unsupported.** Both seeds measure `code`'s untrained recall@1 at 0.2285 (seed 0) and
0.2344 (region-specific seed) — consistent with the surviving receipt (0.2285, both the
2026-09-02 and 2026-09-03 receipts) and far below 0.40. No artefact in the tree backs
0.40; every site carrying that figure is prose. Where traced, the 0.40 figure originates
from a pre-shuffle-fix two-repo holdout (an earlier, since-corrected evaluation setup),
not from any measurement against the current held-out bins.

## Provenance

- Receipt: `/akula-data/csd/receipts/w2c-untrained-baselines-20260903T124940Z.json`
- Code revision: `0026a3d` (`0026a3d62b7330ccbf0e8152a5d1fcfd492ddae7`, not dirty)
- Recorded: 2026-09-03T12:49:40Z
- `SHA256SUMS` covers `measure_w2c.py` and `results.json` in this directory.
