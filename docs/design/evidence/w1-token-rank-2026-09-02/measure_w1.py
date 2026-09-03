"""W1 (design doc row W1, "consequence-3 cheap half"): does the pre-pool token surface of
a mean-pooled contrastive text/vision encoder carry materially more effective rank than the
pooled vector it was trained through?

READ-ONLY. Never writes into the repo or into /akula-data/csd/receipts. Imports
cogsyndelta.regions.pretrain (owned by other agents right now) but does not edit it.

Run: uv run --no-sync python measure_w1.py   (from the CogSynDelta repo root)
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = "/home/kang/code/personal/tzervas/CogSynDelta"
sys.path.insert(0, f"{REPO}/src")

OUT_DIR = Path(
    "/tmp/claude-1000/-home-kang-code-personal-tzervas-shai-stuff/"
    "78dd84c6-dd81-4f47-80a4-c4b31b76e34e/scratchpad/exp-w1"
)
CORPUS = Path("/mnt/fleet-datasets/csd")
TOKENIZER_PATH = "/mnt/fleet-datasets/tritter/gpt2_tokenizer.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFER_BATCH = 64
N_HOLDOUT = 512

VERIFIED_NOTES: list[str] = []
INFERRED_NOTES: list[str] = []
UNRELIABLE_NOTES: list[str] = []


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------
# Import with one retry: other agents are editing regions/pretrain.py right now.
# ---------------------------------------------------------------------------------------
def import_with_retry(modpath: str, attrs: list[str], retries: int = 1, wait: float = 60.0):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            sys.modules.pop(modpath, None)
            mod = importlib.import_module(modpath)
            return tuple(getattr(mod, a) for a in attrs)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < retries:
                log(f"import {modpath} failed ({e!r}); retrying in {wait:.0f}s")
                time.sleep(wait)
    raise RuntimeError(f"failed to import {modpath} after retries") from last_exc


PretrainConfig, build_splits, _tokenize = import_with_retry(
    "cogsyndelta.regions.pretrain", ["PretrainConfig", "build_splits", "_tokenize"]
)
TextEncoder, TextEncoderConfig = import_with_retry(
    "cogsyndelta.regions.text_encoder", ["TextEncoder", "TextEncoderConfig"]
)
from cogsyndelta.eval.benchmark import effective_rank as entropy_effective_rank  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

# vl_latent is optional -- only imported if its checkpoint/eval images are present.
IJEPA = JEPAConfig = None
_decode_split = _to_float = None


def shards(pattern: str, root: Path = CORPUS) -> list[str]:
    return sorted(str(p) for p in root.glob(pattern))


# ---------------------------------------------------------------------------------------
# Region specs. Only pair_columns/shards/extra_sources/seed/checkpoint are needed here --
# encoder shape (dim, depth, n_heads, max_len, vocab_size) is read from EACH checkpoint's
# own saved `config` field, never hardcoded, so this cannot drift from what was trained.
# ---------------------------------------------------------------------------------------
TEXT_REGION_SPECS = {
    "code": dict(
        pair_columns=("docstring", "code"),
        shards=shards("region/code/codesearchnet-python/**/*.parquet"),
        extra_sources=[],
        seed=0,
        steps=8000,
        batch_size=256,
        holdout_pairs=512,
        checkpoint="/akula-data/csd/receipts/code-checkpoints/final.pt",
        production=True,
    ),
    "compress": dict(
        pair_columns=("anchor", "positive"),
        shards=shards("region/compress/all-nli/pair/train*.parquet"),
        extra_sources=[],
        seed=0,
        steps=8000,
        batch_size=256,
        holdout_pairs=512,
        checkpoint="/akula-data/csd/receipts/compress-checkpoints/final.pt",
        production=True,
    ),
    "compress_repo_local": dict(
        pair_columns=("anchor", "positive"),
        shards=shards("region/compress/all-nli/pair/train*.parquet"),
        extra_sources=[],
        seed=2,
        steps=2000,
        batch_size=256,
        holdout_pairs=512,
        checkpoint=f"{REPO}/receipts/compress-checkpoints/final.pt",
        production=False,
        note=(
            "NOT the production checkpoint. receipts/compress-20260902T153612Z.json shows "
            "seed=2, max_len=256, steps=2000 -- a different, earlier/exploratory run over the "
            "same corpus. Different sha256 from the /akula-data compress checkpoint. Measured "
            "as a separate row per the task's explicit instruction to include this path; "
            "excluded from cross-region CKA because its tokenization (max_len 256) is not "
            "guaranteed identical to the production regions (max_len 96) for longer texts."
        ),
    ),
    "retrieve": dict(
        pair_columns=("query", "passage"),
        shards=shards("region/retrieve/fiqa-pairs/train.parquet"),
        extra_sources=[
            {
                "shards": shards("region/retrieve/natural-questions/**/train*.parquet"),
                "columns": ["query", "answer"],
                "limit": 0,
            },
            {
                "shards": shards("region/retrieve/gooaq/**/train*.parquet"),
                "columns": ["question", "answer"],
                "limit": 400_000,
            },
        ],
        seed=0,
        steps=8000,
        batch_size=256,
        holdout_pairs=512,
        checkpoint="/akula-data/csd/receipts/retrieve-checkpoints/final.pt",
        production=True,
    ),
}

CKA_REGIONS = ["code", "compress", "retrieve"]  # shared tokenizer + max_len=96 -> aligned CKA


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def text_encode_from_ids(model: "TextEncoder", ids: torch.Tensor, mask: torch.Tensor, batch: int = INFER_BATCH):
    """Replicate TextEncoder.forward exactly, via direct submodule calls, capturing the
    pre-pool post-norm token matrix [T_valid, D] per item alongside the pooled [D] vector.

    No hooks needed: this IS the forward pass, called explicitly so the pre-pool tensor
    never has to be intercepted. Model code is not touched.
    """
    model.eval()
    pooled_chunks = []
    token_list: list[torch.Tensor] = []
    for i in range(0, ids.size(0), batch):
        b_ids = ids[i : i + batch]
        b_mask = mask[i : i + batch]
        b, t = b_ids.shape
        h = model.embed(b_ids) + model.pos_embed[:, :t]
        for block in model.blocks:
            h = block(h, b_mask)
        h = model.norm(h)  # [B, T, D] pre-pool, post-norm tokens
        mask_f = b_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1e-6)
        pooled = model.proj(pooled)
        pooled_chunks.append(pooled.float().cpu())
        for bi in range(b):
            valid = b_mask[bi].bool()
            token_list.append(h[bi][valid].float().cpu())
    pooled_mat = torch.cat(pooled_chunks, dim=0)
    return pooled_mat, token_list


def pr_effective_rank(x: torch.Tensor) -> float:
    """Participation-ratio effective rank: (sum s_i^2)^2 / sum s_i^4, over singular values
    of the column-centered matrix. Per the task's explicit formula (not the entropy one).
    """
    if x.size(0) < 2:
        return float("nan")
    x = x.float()
    xc = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(xc.to(DEVICE) if DEVICE.type == "cuda" else xc)
    s2 = (s.double()) ** 2
    denom = (s2**2).sum()
    if denom <= 0:
        return 0.0
    num = s2.sum() ** 2
    return float((num / denom).item())


def entropy_eff_rank_full(x: torch.Tensor) -> float:
    """cogsyndelta.eval.benchmark.effective_rank (Shannon entropy of the LINEAR, not
    squared, normalised singular-value spectrum, exponentiated) -- confirmed by grep to be
    the definition behind the design doc's "8.7 of 128" figure. Called with sample=N to
    disable the function's default subsampling so we get the exact number on our set.
    """
    if x.size(0) < 2:
        return float("nan")
    return entropy_effective_rank(x, sample=x.size(0))


def mean_pairwise_cosine(x: torch.Tensor) -> float:
    if x.size(0) < 2:
        return float("nan")
    xn = torch.nn.functional.normalize(x.float(), dim=-1)
    sim = xn @ xn.T
    n = sim.size(0)
    off = ~torch.eye(n, dtype=torch.bool)
    return float(sim[off].mean().item())


def within_item_mean_pairwise_cosine(token_list: list[torch.Tensor]) -> tuple[float, int]:
    vals = []
    skipped = 0
    for tok in token_list:
        if tok.size(0) < 2:
            skipped += 1
            continue
        vals.append(mean_pairwise_cosine(tok))
    if not vals:
        return float("nan"), skipped
    return float(sum(vals) / len(vals)), skipped


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA (Kornblith et al. 2019), feature-space form -- efficient since D << N here."""
    x = x.float()
    y = y.float()
    xc = x - x.mean(dim=0, keepdim=True)
    yc = y - y.mean(dim=0, keepdim=True)
    xty = xc.T @ yc
    hsic = (xty * xty).sum()
    nx = (xc.T @ xc)
    ny = (yc.T @ yc)
    normx = (nx * nx).sum().sqrt()
    normy = (ny * ny).sum().sqrt()
    denom = normx * normy
    if denom <= 0:
        return float("nan")
    return float((hsic / denom).item())


def t_valid_stats(mask_sums: list[int]) -> dict:
    xs = sorted(mask_sums)
    n = len(xs)
    mean = sum(xs) / n
    median = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    return {"mean": mean, "median": median, "max": max(xs), "min": min(xs)}


# ---------------------------------------------------------------------------------------
# Per-region processing
# ---------------------------------------------------------------------------------------
def load_text_models(checkpoint_path: str, seed: int):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg_dict = dict(ckpt["config"])
    enc_cfg = TextEncoderConfig(**cfg_dict)

    torch.manual_seed(seed)  # exact replica of pretrain_region's torch.manual_seed(cfg.seed)
    # placed immediately before construction, matching the production call order, so this
    # reproduces the SAME untrained init the receipt's `untrained_baseline` was measured on.
    model_untrained = TextEncoder(enc_cfg).to(DEVICE)
    model_untrained.eval()

    model_trained = TextEncoder(enc_cfg).to(DEVICE)
    model_trained.load_state_dict(ckpt["model"])
    model_trained.to(DEVICE)
    model_trained.eval()

    sha = sha256_file(checkpoint_path)
    return model_trained, model_untrained, enc_cfg, sha, ckpt.get("step")


def process_text_region(name: str, spec: dict, tok: Tokenizer) -> dict:
    log(f"=== {name} : build_splits ===")
    cfg = PretrainConfig(
        region=name,
        pair_columns=spec["pair_columns"],
        shards=spec["shards"],
        extra_sources=spec["extra_sources"],
        steps=spec["steps"],
        batch_size=spec["batch_size"],
        holdout_pairs=spec["holdout_pairs"],
        seed=spec["seed"],
        tokenizer_path=TOKENIZER_PATH,
    )
    t0 = time.time()
    holdout, _train_pairs, meta = build_splits(cfg)
    log(f"    build_splits done in {time.time() - t0:.1f}s, holdout={len(holdout)}")
    assert len(holdout) == N_HOLDOUT, f"expected {N_HOLDOUT} holdout pairs, got {len(holdout)}"
    anchors = [a for a, _p in holdout]

    log(f"=== {name} : load models from {spec['checkpoint']} ===")
    model_trained, model_untrained, enc_cfg, sha, ckpt_step = load_text_models(
        spec["checkpoint"], spec["seed"]
    )
    max_len = enc_cfg.max_len
    ids, mask = _tokenize(tok, anchors, max_len, DEVICE)

    # Correctness check: manual replication must match TextEncoder.forward exactly.
    with torch.no_grad():
        pooled_manual, _ = text_encode_from_ids(model_trained, ids[:8], mask[:8], batch=8)
        pooled_official = model_trained(ids[:8], mask[:8]).float().cpu()
    max_abs_diff = (pooled_manual - pooled_official).abs().max().item()
    if max_abs_diff > 1e-4:
        UNRELIABLE_NOTES.append(
            f"{name}: manual submodule replication of forward() diverged from model(ids,mask) "
            f"by max_abs={max_abs_diff:.2e} -- pre-pool token capture may not equal the real "
            f"eval path."
        )
    else:
        VERIFIED_NOTES.append(
            f"{name}: manual forward replication matches model(ids,mask) to max_abs="
            f"{max_abs_diff:.2e} (8-item check)."
        )

    out = {"regions": {}}
    for tag, model in (("trained", model_trained), ("untrained", model_untrained)):
        log(f"    encoding ({tag})...")
        pooled, tokens = text_encode_from_ids(model, ids, mask, batch=INFER_BATCH)
        mask_sums = [t.size(0) for t in tokens]
        n_short = sum(1 for m in mask_sums if m < 2)

        token_cat = torch.cat([t for t in tokens if t.size(0) >= 1], dim=0)
        per_item_pr = [pr_effective_rank(t) for t in tokens if t.size(0) >= 2]
        per_item_pr_mean = (
            float(sum(per_item_pr) / len(per_item_pr)) if per_item_pr else float("nan")
        )
        within_cos, skipped = within_item_mean_pairwise_cosine(tokens)

        metrics = {
            "pooled_pr_rank": pr_effective_rank(pooled),
            "pooled_entropy_rank": entropy_eff_rank_full(pooled),
            "token_global_pr_rank": pr_effective_rank(token_cat),
            "token_global_entropy_rank": entropy_eff_rank_full(token_cat),
            "token_per_item_pr_rank_mean": per_item_pr_mean,
            "token_per_item_pr_rank_n_items_used": len(per_item_pr),
            "pooled_mean_pairwise_cosine": mean_pairwise_cosine(pooled),
            "token_within_item_mean_pairwise_cosine": within_cos,
            "token_within_item_cosine_items_skipped_lt2": skipped,
            "n_short_items_lt2_tokens": n_short,
            "t_valid": t_valid_stats(mask_sums),
            "n_items": pooled.size(0),
            "n_tokens_total": token_cat.size(0),
        }
        out["regions"][tag] = metrics
        # keep tensors for CKA use (production regions only)
        if tag == "trained":
            out["_pooled_trained"] = pooled
            out["_tokens_trained"] = tokens
        else:
            out["_pooled_untrained"] = pooled
            out["_tokens_untrained"] = tokens

    out["dim"] = enc_cfg.dim
    out["max_len"] = max_len
    out["checkpoint"] = spec["checkpoint"]
    out["checkpoint_sha256"] = sha
    out["checkpoint_step"] = ckpt_step
    out["production"] = spec.get("production", True)
    if "note" in spec:
        out["note"] = spec["note"]
    out["_anchors"] = anchors  # kept for building the shared CKA set
    out["_ids_mask_maxlen"] = max_len
    return out


# ---------------------------------------------------------------------------------------
# vl_latent (optional)
# ---------------------------------------------------------------------------------------
def try_load_vl():
    global IJEPA, JEPAConfig, _decode_split, _to_float
    ckpt_path = Path("/akula-data/csd/receipts/vl_latent-checkpoints/step-8000.pt")
    valid_shards = shards("vl/tiny-imagenet/data/valid-*.parquet", root=CORPUS)
    if not ckpt_path.is_file() or not valid_shards:
        return None, None
    (IJEPA, JEPAConfig) = import_with_retry("cogsyndelta.model.vl_jepa", ["IJEPA", "JEPAConfig"])
    (_decode_split, _to_float) = import_with_retry(
        "cogsyndelta.regions.vl_pretrain", ["_decode_split", "_to_float"]
    )
    return ckpt_path, valid_shards


@torch.no_grad()
def vl_encode_from_images(encoder, images_f: torch.Tensor, batch: int = INFER_BATCH):
    """Direct submodule call on ViTEncoder (patch_embed -> +pos -> blocks -> norm), the
    exact computation IJEPA.encode()/.embed() perform, captured before the mean-pool.
    """
    encoder.eval()
    pooled_chunks = []
    token_list = []
    for i in range(0, images_f.size(0), batch):
        x = images_f[i : i + batch]
        h = encoder.patch_embed(x) + encoder.pos_embed
        for block in encoder.blocks:
            h = block(h)
        h = encoder.norm(h)  # [B, N, D] pre-pool patch tokens (no padding: every patch valid)
        pooled = h.mean(dim=1)
        pooled_chunks.append(pooled.float().cpu())
        for bi in range(h.size(0)):
            token_list.append(h[bi].float().cpu())
    return torch.cat(pooled_chunks, dim=0), token_list


def process_vl_latent(ckpt_path: Path, valid_shards: list[str]) -> dict:
    log("=== vl_latent : decode held-out images ===")
    cache_dir = Path("/tmp/claude-1000/-home-kang-code-personal-tzervas-shai-stuff/"
                      "78dd84c6-dd81-4f47-80a4-c4b31b76e34e/scratchpad/exp-w1/vl-cache")
    x, _y = _decode_split(valid_shards, "image", "label", 64, N_HOLDOUT, cache_dir)
    assert x.size(0) == N_HOLDOUT, f"expected {N_HOLDOUT} images, got {x.size(0)}"
    images_f = _to_float(x, DEVICE)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    jcfg = JEPAConfig(**ckpt["config"])

    torch.manual_seed(0)  # VLPretrainConfig default seed, matches vl_pretrain's untrained baseline
    model_untrained = IJEPA(jcfg).to(DEVICE)
    model_untrained.eval()

    model_trained = IJEPA(jcfg).to(DEVICE)
    model_trained.load_state_dict(ckpt["model"])
    model_trained.to(DEVICE)
    model_trained.eval()

    sha = sha256_file(str(ckpt_path))

    # correctness check against IJEPA.encode() (== target_encoder.embed())
    with torch.no_grad():
        pooled_manual, _ = vl_encode_from_images(model_trained.target_encoder, images_f[:8], batch=8)
        pooled_official = model_trained.encode(images_f[:8]).float().cpu()
    max_abs_diff = (pooled_manual - pooled_official).abs().max().item()
    if max_abs_diff > 1e-4:
        UNRELIABLE_NOTES.append(
            f"vl_latent: manual ViTEncoder replication diverged from IJEPA.encode() by "
            f"max_abs={max_abs_diff:.2e}."
        )
    else:
        VERIFIED_NOTES.append(
            f"vl_latent: manual ViTEncoder replication matches IJEPA.encode() to max_abs="
            f"{max_abs_diff:.2e} (8-item check)."
        )

    out = {"regions": {}}
    for tag, model in (("trained", model_trained), ("untrained", model_untrained)):
        log(f"    encoding vl_latent ({tag})...")
        pooled, tokens = vl_encode_from_images(model.target_encoder, images_f, batch=INFER_BATCH)
        mask_sums = [t.size(0) for t in tokens]  # constant = n_patches, no padding
        token_cat = torch.cat(tokens, dim=0)
        per_item_pr = [pr_effective_rank(t) for t in tokens]
        within_cos, skipped = within_item_mean_pairwise_cosine(tokens)
        metrics = {
            "pooled_pr_rank": pr_effective_rank(pooled),
            "pooled_entropy_rank": entropy_eff_rank_full(pooled),
            "token_global_pr_rank": pr_effective_rank(token_cat),
            "token_global_entropy_rank": entropy_eff_rank_full(token_cat),
            "token_per_item_pr_rank_mean": float(sum(per_item_pr) / len(per_item_pr)),
            "token_per_item_pr_rank_n_items_used": len(per_item_pr),
            "pooled_mean_pairwise_cosine": mean_pairwise_cosine(pooled),
            "token_within_item_mean_pairwise_cosine": within_cos,
            "token_within_item_cosine_items_skipped_lt2": skipped,
            "n_short_items_lt2_tokens": 0,
            "t_valid": t_valid_stats(mask_sums),
            "n_items": pooled.size(0),
            "n_tokens_total": token_cat.size(0),
        }
        out["regions"][tag] = metrics
    out["dim"] = jcfg.dim
    out["n_patches"] = (jcfg.image_size // jcfg.patch_size) ** 2
    out["checkpoint"] = str(ckpt_path)
    out["checkpoint_sha256"] = sha
    out["checkpoint_step"] = ckpt["step"]
    out["production"] = True
    out["note"] = (
        "Vision region: token = patch (all patches valid, no padding). Checkpoint on disk "
        "carries only {step, model, config} -- fewer keys than _checkpoint_payload's current "
        "schema (no opt/rng state), i.e. it predates the richer resumable-checkpoint format; "
        "irrelevant to this read-only inference measurement. Not included in the text "
        "cross-region CKA matrix -- different modality, no shared raw-text input exists."
    )
    return out


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"regions": {}, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    region_data: dict[str, dict] = {}

    tok = Tokenizer.from_file(TOKENIZER_PATH)
    VERIFIED_NOTES.append(f"tokenizer loaded from {TOKENIZER_PATH}, vocab_size={tok.get_vocab_size()}")

    for name, spec in TEXT_REGION_SPECS.items():
        try:
            data = process_text_region(name, spec, tok)
            region_data[name] = data
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            UNRELIABLE_NOTES.append(f"{name}: FAILED -- {e!r}\n{tb}")
            log(f"!!! {name} failed: {e!r}")

    # vl_latent, best-effort
    try:
        ckpt_path, valid_shards = try_load_vl()
        if ckpt_path is None:
            INFERRED_NOTES.append("vl_latent: checkpoint or eval images not found on this host; skipped.")
        else:
            vl_data = process_vl_latent(ckpt_path, valid_shards)
            region_data["vl_latent"] = vl_data
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        UNRELIABLE_NOTES.append(f"vl_latent: FAILED -- {e!r}\n{tb}")
        log(f"!!! vl_latent failed: {e!r}")

    # ---- write per-region metrics (strip tensors) ----
    for name, data in region_data.items():
        clean = {k: v for k, v in data.items() if not k.startswith("_")}
        results["regions"][name] = clean

    # ---- cross-region CKA on a shared mixed 512-item text set ----
    cka_names = [n for n in CKA_REGIONS if n in region_data]
    if len(cka_names) >= 2:
        log(f"=== cross-region CKA on {cka_names} ===")
        try:
            per_n = N_HOLDOUT // len(cka_names)
            remainder = N_HOLDOUT - per_n * len(cka_names)
            shared_texts: list[str] = []
            for i, n in enumerate(cka_names):
                take = per_n + (1 if i < remainder else 0)
                shared_texts.extend(region_data[n]["_anchors"][:take])
            assert len(shared_texts) == N_HOLDOUT

            max_len = max(region_data[n]["_ids_mask_maxlen"] for n in cka_names)
            assert all(region_data[n]["_ids_mask_maxlen"] == max_len for n in cka_names), (
                "cka regions do not share max_len; aligned token CKA would be invalid"
            )
            VERIFIED_NOTES.append(
                f"CKA regions {cka_names} share tokenizer={TOKENIZER_PATH} and max_len={max_len}, "
                f"so token-level CKA on aligned (item, position) rows is valid."
            )
            ids, mask = _tokenize(tok, shared_texts, max_len, DEVICE)

            cka_pooled = {"trained": {}, "untrained": {}}
            cka_tokens = {"trained": {}, "untrained": {}}
            for n in cka_names:
                ckpt = torch.load(TEXT_REGION_SPECS[n]["checkpoint"], map_location="cpu", weights_only=True)
                enc_cfg = TextEncoderConfig(**dict(ckpt["config"]))
                torch.manual_seed(TEXT_REGION_SPECS[n]["seed"])
                m_un = TextEncoder(enc_cfg).to(DEVICE)
                m_un.eval()
                m_tr = TextEncoder(enc_cfg).to(DEVICE)
                m_tr.load_state_dict(ckpt["model"])
                m_tr.to(DEVICE)
                m_tr.eval()
                for tag, m in (("trained", m_tr), ("untrained", m_un)):
                    pooled, tokens = text_encode_from_ids(m, ids, mask, batch=INFER_BATCH)
                    cka_pooled[tag][n] = pooled
                    cka_tokens[tag][n] = torch.cat(tokens, dim=0)  # aligned rows across regions

            cka_matrices = {"pooled": {}, "token_aligned": {}}
            for tag in ("trained", "untrained"):
                mat_p = {}
                mat_t = {}
                for a in cka_names:
                    for b in cka_names:
                        key = f"{a}__{b}"
                        mat_p[key] = linear_cka(cka_pooled[tag][a], cka_pooled[tag][b])
                        mat_t[key] = linear_cka(cka_tokens[tag][a], cka_tokens[tag][b])
                cka_matrices["pooled"][tag] = mat_p
                cka_matrices["token_aligned"][tag] = mat_t
            results["cross_region_cka"] = cka_matrices
            results["cka_shared_set_composition"] = {
                n: (N_HOLDOUT // len(cka_names)) + (1 if i < remainder else 0)
                for i, n in enumerate(cka_names)
            }
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            UNRELIABLE_NOTES.append(f"cross-region CKA: FAILED -- {e!r}\n{tb}")
            log(f"!!! CKA failed: {e!r}")
    else:
        INFERRED_NOTES.append("cross-region CKA: fewer than 2 production text regions available; skipped.")

    # ---- decision rule ----
    verdicts = {}
    for name, data in region_data.items():
        m = data["regions"].get("trained")
        if not m:
            continue
        pooled_pr = m["pooled_pr_rank"]
        token_pr = m["token_global_pr_rank"]
        ratio = token_pr / pooled_pr if pooled_pr else float("nan")
        if ratio >= 2.0:
            verdict = "BET ALIVE for this region (token >= 2x pooled): no retrain required for phase 2"
        elif ratio <= 1.5:
            verdict = "BET DEAD for this region (token approx pooled, <=1.5x): retrain before interconnect work"
        else:
            verdict = "AMBIGUOUS (between 1.5x and 2x pooled) -- neither pre-committed threshold met"
        flags = []
        if token_pr < 8:
            flags.append(f"FLAG: token-level effective rank {token_pr:.2f} < 8")
        verdicts[name] = {
            "pooled_pr_rank": pooled_pr,
            "token_global_pr_rank": token_pr,
            "ratio": ratio,
            "verdict": verdict,
            "flags": flags,
        }
    # CKA flags
    if "cross_region_cka" in results:
        for key, val in results["cross_region_cka"]["pooled"]["trained"].items():
            a, b = key.split("__")
            if a != b and val > 0.90:
                verdicts.setdefault("_cka_flags", []).append(f"FLAG: pooled CKA({a},{b})={val:.3f} > 0.90")
    results["verdicts"] = verdicts
    results["verified_notes"] = VERIFIED_NOTES
    results["inferred_notes"] = INFERRED_NOTES
    results["unreliable_notes"] = UNRELIABLE_NOTES

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote {OUT_DIR / 'results.json'}")

    return results, region_data


if __name__ == "__main__":
    main()
