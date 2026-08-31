#!/usr/bin/env python3
"""Run PoC CUDA tests on a live GPU. No pytest required. Receipt JSON to --out."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    out: dict = {"ok": False, "cuda": False, "device": None, "tests": []}
    try:
        import torch
    except ImportError as exc:
        out["error"] = f"no torch: {exc}"
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return 1
    out["cuda"] = bool(torch.cuda.is_available())
    if out["cuda"]:
        out["device"] = torch.cuda.get_device_name(0)
        out["torch"] = torch.__version__
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    import tests.test_poc_cuda as m

    failed = 0
    for name, fn in vars(m).items():
        if not name.startswith("test_") or not callable(fn):
            continue
        rec: dict = {"name": name, "ok": False}
        try:
            fn()
            rec["ok"] = True
        except Exception as exc:  # noqa: BLE001
            failed += 1
            rec["error"] = str(exc)
            rec["tb"] = traceback.format_exc()[-1500:]
        out["tests"].append(rec)
    out["ok"] = failed == 0 and bool(out["cuda"]) and bool(out["tests"])
    out["seconds"] = round(time.time() - t0, 2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2)[:8000])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
