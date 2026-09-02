#!/usr/bin/env python3
"""Measure what a CSD training step actually costs, so optimisations can be judged.

WHY THIS EXISTS
P9 asks whether bf16, torch.compile, gradient accumulation and friends are worth adding.
None of that is answerable without a decomposition of the current step: a technique that
halves GPU matmul time is worth nothing if the step is dominated by a synchronous
single-threaded CPU tokenizer, and the receipts cannot tell you which it is -- they record
`elapsed_s` for the whole run and nothing about where it went.

This is a MEASUREMENT tool, not part of the harness. It never writes to the receipt
directory and never touches a checkpoint. It is safe to run while a training job is
executing: everything except `--gpu-probe` is CPU- and HTTP-only, and the probe caps its
own share of the card so it cannot starve a live run into an OOM.

FOUR THINGS IT MEASURES
1. `receipts`   -- steps/sec, ms/step and pair throughput derived from real receipts.
2. `prometheus` -- peak GPU memory inside each receipt's own wall-clock window, minus the
                   desktop idle floor, so the number is what TRAINING used.
3. `tokenizer`  -- CPU cost of tokenizing one batch, serial (what the harness does today)
                   against `encode_batch`, plus padded-width distribution and how much of
                   the 50,257-token GPT-2 vocabulary a region actually touches.
4. `gpu-probe`  -- optional, opt-in: peak allocated VRAM and step time against batch size,
                   fp32 versus bf16 autocast, on synthetic ids. Synthetic because the point
                   is to isolate the GPU: mixing the tokenizer back in measures the thing
                   part 3 already measured.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import statistics
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

CORPUS = Path("/mnt/fleet-datasets/csd/region")
TOKENIZER = "/mnt/fleet-datasets/tritter/gpt2_tokenizer.json"
PROM = "http://192.168.1.170:8428"
PROM_SELECTOR = 'akula_gpu_memory_used_mib{instance="192.168.1.98:9108"}'

# One representative shard per region, with the columns the runner pairs. Deliberately one
# shard and not the glob: this measures per-batch cost, which does not vary across shards
# of the same corpus, and reading 4 shards to learn the same number is wasted IO on a
# machine that is currently training.
REGION_SAMPLES: dict[str, tuple[str, tuple[str, str], str | None]] = {
    "code": (
        "code/codesearchnet-python/data/train-00000-of-00004-ee77a7de79eb2ab2.parquet",
        ("docstring", "code"),
        "repo",
    ),
    "compress": (
        "compress/all-nli/pair/train-00000-of-00001.parquet",
        ("anchor", "positive"),
        None,
    ),
    "retrieve": (
        "retrieve/fiqa-pairs/train.parquet",
        ("query", "passage"),
        None,
    ),
    "retrieve-gooaq": (
        "retrieve/gooaq/pair/train-00000-of-00002.parquet",
        ("question", "answer"),
        None,
    ),
}


def _iso(seconds: float) -> str:
    return dt.datetime.fromtimestamp(seconds, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(stamp: str) -> float:
    return dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.UTC).timestamp()


# --------------------------------------------------------------------------- receipts


def from_receipts(receipt_dir: Path) -> list[dict[str, Any]]:
    """Derive per-run step time and throughput from pretrain receipts.

    A receipt records total `elapsed_s` and a step count, never a step time. Dividing them
    is the whole trick, and it is worth doing once in code rather than in each analysis,
    because the two schemas name the duration differently (`elapsed_s` for text regions,
    `seconds` for the visual one) and an analysis that knows only one of them silently
    drops half the fleet.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(receipt_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cfg = data.get("config") or {}
        steps = cfg.get("steps")
        batch = cfg.get("batch_size")
        elapsed = data.get("elapsed_s", data.get("seconds"))
        if not steps or not elapsed:
            continue
        params = data.get("parameters")
        row = {
            "receipt": path.name,
            "region": data.get("region", "?"),
            "steps": steps,
            "batch": batch,
            "elapsed_s": elapsed,
            "steps_per_s": round(steps / elapsed, 3),
            "ms_per_step": round(1000.0 * elapsed / steps, 1),
            "parameters": params,
        }
        if batch:
            row["pairs_per_s"] = round(steps * batch / elapsed, 1)
            # Two encoder forwards per step -- anchor and positive -- so the sequence
            # count the GPU sees is twice the pair count. Recording both stops a later
            # comparison against a single-tower baseline being off by exactly 2x.
            row["sequences_per_s"] = round(2 * steps * batch / elapsed, 1)
        started = data.get("recorded") or data.get("started_utc")
        if started:
            finished = _parse_iso(started)
            # A text receipt stamps the time it was WRITTEN, i.e. the end; the visual one
            # stamps the start. Normalise both to a [start, end] window so the Prometheus
            # lookup below asks about the right minutes.
            if "elapsed_s" in data:
                row["window"] = (_iso(finished - elapsed), started)
            else:
                row["window"] = (started, _iso(finished + elapsed))
        rows.append(row)
    return rows


# ------------------------------------------------------------------------- prometheus


def prom_range(expr: str, start: str, end: str, step: str = "15s") -> list[tuple[float, float]]:
    """Fetch one metric series over a window. Empty list when nothing matched."""
    url = f"{PROM}/api/v1/query_range?" + urllib.parse.urlencode(
        {"query": expr, "start": start, "end": end, "step": step}
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    series = payload.get("data", {}).get("result", [])
    if not series:
        return []
    return [(float(t), float(v)) for t, v in series[0]["values"]]


def vram_for_windows(rows: list[dict[str, Any]], idle_floor_mib: float) -> list[dict[str, Any]]:
    """Peak GPU memory inside each run's own window, net of the idle desktop floor.

    The floor is not cosmetic. akula-prime runs a KDE desktop on the same card, so the
    raw metric never falls below ~922 MiB and reporting it as training memory overstates
    the run by roughly a fifth of what it actually used.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        window = row.get("window")
        if not window:
            continue
        values = [v for _, v in prom_range(PROM_SELECTOR, window[0], window[1])]
        if not values:
            out.append({**{k: row[k] for k in ("region", "receipt")}, "samples": 0})
            continue
        out.append(
            {
                "region": row["region"],
                "receipt": row["receipt"],
                "window": window,
                "samples": len(values),
                "peak_total_mib": max(values),
                "median_total_mib": statistics.median(values),
                "peak_training_mib": round(max(values) - idle_floor_mib, 1),
            }
        )
    return out


# -------------------------------------------------------------------------- tokenizer


def _load_pairs(path: Path, columns: tuple[str, str], limit: int) -> list[tuple[str, str]]:
    import pyarrow.parquet as pq

    left, right = columns
    pairs: list[tuple[str, str]] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=2048, columns=list(columns)):
        a_col = batch.column(left).to_pylist()
        b_col = batch.column(right).to_pylist()
        for a, b in zip(a_col, b_col, strict=True):
            if a and b and a.strip() and b.strip():
                pairs.append((a, b))
                if len(pairs) >= limit:
                    return pairs
    return pairs


def tokenizer_cost(batch_size: int, max_len: int, batches: int, vocab_texts: int) -> dict[str, Any]:
    """Cost of turning one training batch into ids, and how much vocabulary it touches.

    Serial versus `encode_batch` because that is the decision: `regions/pretrain.py`
    tokenizes with a Python list comprehension over single `encode` calls, which holds the
    GIL-free Rust tokenizer to one core while the GPU waits.

    Vocabulary coverage is here rather than in a separate tool because it shares the
    expensive part (encoding a lot of real text) and it is the denominator of the project's
    own thesis metric: 80.3% of the encoder's 16.02M parameters are a 50,257-row embedding
    table, so how many of those rows a region ever indexes decides whether that count is
    capability or dead weight.
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER)
    report: dict[str, Any] = {}
    for region, (rel, columns, group_col) in REGION_SAMPLES.items():
        path = CORPUS / rel
        if not path.exists():
            report[region] = {"error": f"missing {path}"}
            continue
        pairs = _load_pairs(path, columns, batch_size * batches)
        usable = len(pairs) // batch_size
        if usable == 0:
            report[region] = {"error": f"only {len(pairs)} pairs"}
            continue

        serial_s = 0.0
        parallel_s = 0.0
        widths: list[int] = []
        for i in range(usable):
            chunk = pairs[i * batch_size : (i + 1) * batch_size]
            for side in (0, 1):
                texts = [c[side] for c in chunk]
                t0 = time.perf_counter()
                encoded = [tok.encode(t).ids[:max_len] for t in texts]
                serial_s += time.perf_counter() - t0
                widths.append(max(1, *(len(e) for e in encoded)))
                t0 = time.perf_counter()
                tok.encode_batch(texts)
                parallel_s += time.perf_counter() - t0

        # Vocabulary coverage over a wider sample than the timing loop can afford.
        wide = _load_pairs(path, columns, vocab_texts)
        seen: collections.Counter[int] = collections.Counter()
        flat = [t for pair in wide for t in pair]
        for i in range(0, len(flat), 4096):
            for enc in tok.encode_batch(flat[i : i + 4096]):
                seen.update(enc.ids[:max_len])

        report[region] = {
            "batches_timed": usable,
            "serial_ms_per_step": round(1000.0 * serial_s / usable, 2),
            "encode_batch_ms_per_step": round(1000.0 * parallel_s / usable, 2),
            "parallel_speedup": round(serial_s / parallel_s, 2) if parallel_s else None,
            "mean_padded_width": round(statistics.mean(widths), 1),
            "max_padded_width": max(widths),
            "vocab_texts_scanned": len(flat),
            "distinct_tokens": len(seen),
            "vocab_coverage": round(len(seen) / 50257, 4),
            # A row indexed once in 200k texts is noise, not vocabulary. The 99.9%
            # coverage count is the honest size of a pruned embedding table.
            "tokens_covering_99.9pct_of_uses": _cover(seen, 0.999),
            "tokens_covering_99.99pct_of_uses": _cover(seen, 0.9999),
        }
        if group_col:
            report[region]["batch_homogeneity"] = _homogeneity(path, group_col, batch_size)
    return report


def _cover(counts: collections.Counter[int], fraction: float) -> int:
    """How many distinct tokens account for `fraction` of all token occurrences."""
    total = sum(counts.values())
    running = 0
    for i, (_, n) in enumerate(counts.most_common(), start=1):
        running += n
        if running >= fraction * total:
            return i
    return len(counts)


def _homogeneity(path: Path, column: str, batch_size: int, windows: int = 32) -> dict[str, Any]:
    """Distinct values of a grouping column inside consecutive batch-sized windows.

    This is the question "what are the InfoNCE negatives actually made of". The harness
    slices `train_pairs[lo:lo+batch]` from corpus order, and only shuffles when a region
    declares extra sources -- so a single-source region trains on whatever contiguity the
    parquet happens to have.
    """
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    values: list[Any] = []
    for batch in parquet.iter_batches(batch_size=2048, columns=[column]):
        values.extend(batch.column(column).to_pylist())
        if len(values) >= batch_size * windows:
            break
    counts = [
        len(set(values[i * batch_size : (i + 1) * batch_size]))
        for i in range(min(windows, len(values) // batch_size))
    ]
    return {
        "column": column,
        "batch_size": batch_size,
        "windows": len(counts),
        "distinct_per_window": counts,
        "median_distinct": statistics.median(counts) if counts else 0,
        "distinct_in_first_window": counts[0] if counts else 0,
    }


# -------------------------------------------------------------------------- gpu probe


def _train_step(model, opt, ids, mask, autocast) -> None:
    """One full training step: two towers, InfoNCE, clip, update.

    A module-level function rather than a closure inside the sweep. A closure would
    capture the loop's model and optimiser by reference, which is both a lint error and a
    real hazard here -- the sweep rebinds them per configuration and deletes them in a
    `finally`, so a stale capture would time the wrong model or fail on a freed one.
    """
    import torch

    from cogsyndelta.regions.text_encoder import info_nce

    with autocast:
        loss, _ = info_nce(model(ids, mask), model(ids, mask))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def gpu_probe(
    batches: list[int], max_len: int, steps: int, warmup: int, fraction: float
) -> dict[str, Any]:
    """Peak allocated VRAM and step time against batch size, fp32 versus bf16.

    Synthetic ids, all positions real: this is the worst case the harness can produce
    (`_tokenize` pads to the longest item in the batch, and `code` already hits the 96
    ceiling on every batch) and it isolates the GPU from the tokenizer.

    `fraction` caps this process's share of the card through the caching allocator, so a
    probe that asks for too large a batch OOMs ITSELF rather than pushing a concurrently
    running training job over the edge. That is the entire reason the cap exists; do not
    raise it to "make the big batch fit".
    """
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

    if not torch.cuda.is_available():
        return {"error": "no cuda"}
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    device = torch.device("cuda")
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20

    results = []
    for dtype_name in ("fp32", "bf16"):
        for batch in batches:
            torch.manual_seed(0)
            cfg = TextEncoderConfig(vocab_size=50257, dim=256, depth=4, n_heads=4, max_len=max_len)
            model = TextEncoder(cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            ids = torch.randint(0, 50257, (batch, max_len), device=device)
            mask = torch.ones(batch, max_len, dtype=torch.long, device=device)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype_name == "bf16")

            row: dict[str, Any] = {"dtype": dtype_name, "batch": batch, "max_len": max_len}
            try:
                for _ in range(warmup):
                    _train_step(model, opt, ids, mask, autocast)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(steps):
                    _train_step(model, opt, ids, mask, autocast)
                torch.cuda.synchronize()
                row["ms_per_step"] = round(1000.0 * (time.perf_counter() - t0) / steps, 2)
                row["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                row["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
            except torch.OutOfMemoryError:
                row["error"] = f"OOM under the self-imposed {fraction:.2f} cap"
            finally:
                del model, opt, ids, mask
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            results.append(row)
            print(f"    {row}", flush=True)

    return {
        "device_total_mib": round(total_mib, 1),
        "self_cap_mib": round(total_mib * fraction, 1),
        "note": (
            "Timings were taken on a card that was concurrently running a real training "
            "job. Treat the RATIOS between rows as sound and the absolute ms as an upper "
            "bound; the memory figures are per-process and unaffected by the neighbour."
        ),
        "rows": results,
    }


# ------------------------------------------------------------------------------- main


def main() -> int:
    """Run the requested measurements and print (optionally write) a JSON report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--receipts", default="/akula-data/csd/receipts")
    ap.add_argument("--idle-floor-mib", type=float, default=922.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--tokenizer-batches", type=int, default=40)
    ap.add_argument("--vocab-texts", type=int, default=100_000)
    ap.add_argument("--skip-prometheus", action="store_true")
    ap.add_argument("--skip-tokenizer", action="store_true")
    ap.add_argument(
        "--gpu-probe",
        action="store_true",
        help="opt-in: allocates on GPU 0. Safe beside a live run, but not free.",
    )
    ap.add_argument("--probe-batches", default="256,512,1024,2048")
    ap.add_argument("--probe-steps", type=int, default=10)
    ap.add_argument("--probe-warmup", type=int, default=3)
    ap.add_argument(
        "--probe-fraction",
        type=float,
        default=0.35,
        help="hard cap on this process's share of the card, so a probe cannot OOM a "
        "concurrently running training job",
    )
    ap.add_argument("--json", default="", help="write the full report here")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    report: dict[str, Any] = {"measured_utc": _iso(time.time())}

    print("== receipts ==", flush=True)
    rows = from_receipts(Path(args.receipts))
    report["receipts"] = rows
    for row in rows:
        print(
            f"  {row['region']:<10} {row['steps']:>5} steps  {row['elapsed_s']:>7.1f}s  "
            f"{row['ms_per_step']:>7.1f} ms/step  {row['steps_per_s']:>6.2f} steps/s  "
            f"{row.get('pairs_per_s', 0):>9.1f} pairs/s",
            flush=True,
        )

    if not args.skip_prometheus:
        print("\n== peak gpu memory, per run window ==", flush=True)
        try:
            vram = vram_for_windows(rows, args.idle_floor_mib)
        except OSError as exc:
            vram = [{"error": f"{type(exc).__name__}: {exc}"}]
        report["vram"] = vram
        for row in vram:
            if "peak_training_mib" not in row:
                print(f"  {row}", flush=True)
                continue
            print(
                f"  {row['region']:<10} peak {row['peak_total_mib']:>7.0f} MiB total, "
                f"{row['peak_training_mib']:>7.0f} MiB training "
                f"({row['samples']} samples)",
                flush=True,
            )

    if not args.skip_tokenizer:
        print("\n== tokenizer cost per step (CPU) ==", flush=True)
        tok_report = tokenizer_cost(
            args.batch, args.max_len, args.tokenizer_batches, args.vocab_texts
        )
        report["tokenizer"] = tok_report
        for region, row in tok_report.items():
            if "error" in row:
                print(f"  {region:<15} {row['error']}", flush=True)
                continue
            print(
                f"  {region:<15} serial {row['serial_ms_per_step']:>7.2f} ms  "
                f"encode_batch {row['encode_batch_ms_per_step']:>6.2f} ms  "
                f"({row['parallel_speedup']}x)  width~{row['mean_padded_width']:>5.1f}  "
                f"vocab {row['distinct_tokens']:>6} / 50257",
                flush=True,
            )

    if args.gpu_probe:
        print("\n== gpu probe (synthetic ids) ==", flush=True)
        report["gpu_probe"] = gpu_probe(
            [int(b) for b in args.probe_batches.split(",") if b],
            args.max_len,
            args.probe_steps,
            args.probe_warmup,
            args.probe_fraction,
        )

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
