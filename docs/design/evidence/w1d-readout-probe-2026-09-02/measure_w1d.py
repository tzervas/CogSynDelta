"""W1d -- matched read-out probe, the DECIDING step for W1 (design doc §4.1, DEC-35).

READ-ONLY. Never writes into the repo, into src/, or into /akula-data/csd/receipts.
Imports cogsyndelta.regions.pretrain / text_encoder / model.vl_jepa / regions.vl_pretrain
but never edits them.

Spec (docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md, §4.1 table, W1d row; DEC-35):
  Train the SAME small cross-attention read-out, identical params/steps/items, in three
  arms that differ ONLY in which activation surface it attends over:
    (a) tokens()            -- final-block, post-norm, pre-pool per-token activations
    (b) pool() broadcast    -- the pooled vector, repeated to the same T, so the read-out
                                receives zero positionally-differentiated information
    (c) penultimate-block   -- same as (a) but one block earlier (answers old-W7: where
                                to insert the retrain's L_token term)
  Rule (names no rank definition): arm(a) beats arm(b) by >= 2 points (region's own
  receipt metric, in percentage points) => W1 OVERTURNED (retrain cancelled). Arm(a)
  within 2 points of arm(b) => W1 CONFIRMED, provisional status lifted, W4/W7 proceed.
  Verify-by-failing: run arm(a) against arm(a) (different read-out init seed, identical
  data) and confirm the SAME rule reports CONFIRMED; if not, the instrument is too noisy.
  Also report: within-item vs between-item token variance; PR-rank and entropy-rank on
  the probe's own input activations; the penultimate arm's delta (selects the retrain's
  insertion point). Probe trained on a TRAIN split disjoint from the 512-item eval set
  (the "DEV half" instruction of §2.7.8, applied here as: never fit the read-out on the
  items that produce the decision).

Run: uv run --no-sync python measure_w1d.py   (from the CogSynDelta repo root)
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
import torch.nn as nn
import torch.nn.functional as F

REPO = "/home/kang/code/personal/tzervas/CogSynDelta"
sys.path.insert(0, f"{REPO}/src")

OUT_DIR = Path(
    "/tmp/claude-1000/-home-kang-code-personal-tzervas-shai-stuff/"
    "78dd84c6-dd81-4f47-80a4-c4b31b76e34e/scratchpad/exp-w1d"
)
CACHE_DIR = OUT_DIR / "vl-cache"
CORPUS = Path("/mnt/fleet-datasets/csd")
TOKENIZER_PATH = "/mnt/fleet-datasets/tritter/gpt2_tokenizer.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFER_BATCH = 64          # frozen-encoder forward batch, matches measure_w1.py
N_HOLDOUT = 512            # decision-set size, MUST match production build_splits
N_TRAIN_POOL = 4096        # disjoint train slice for fitting the read-out ("DEV half")

READOUT_STEPS = 600
READOUT_BATCH_TEXT = 256   # matches production InfoNCE batch size (comparable difficulty)
READOUT_BATCH_VL = 128
READOUT_LR = 2e-3
READOUT_WARMUP = 60
READOUT_WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

SEED_MAIN = 0              # read-out init seed for arms (a)/(b)/(c), and data batch order
SEED_CONTROL = 1           # different read-out init seed, SAME data -- verify-by-failing
DATA_ORDER_SEED = 12345    # shared batch sampling order across arms within a region

VL_SHUFFLE_SEED = 0            # tiny-imagenet parquets are CLASS-GROUPED, EXACTLY (500
                                # contiguous rows/class train, 50/class valid; confirmed by
                                # direct parquet read: first 512 valid rows span 11 of 200
                                # classes, first 4096 train rows span 9, and the file is
                                # sorted ascending by label). "First N in file order" (W1's
                                # own method, fine for rank stats, which do not care about
                                # label balance) is unusable for a classification probe: a
                                # prefix of ANY size short of the whole file covers only a
                                # contiguous class range, capping the read-out's ceiling
                                # regardless of arm. Fix: stratified_decode() (below) reads
                                # only the label column first, then decodes a seeded,
                                # per-class-balanced sample directly -- no full-split
                                # on-disk cache, which does not fit this host's /tmp
                                # partition (~760MB free; the full 100k-image train split
                                # alone is 1.2GB uint8). See deviation_notes.

RULE_THRESHOLD_POINTS = 2.0  # pre-committed in DEC-35 / §4.1 W1d row, unchanged since rev 2

VERIFIED_NOTES: list[str] = []
INFERRED_NOTES: list[str] = []
UNRELIABLE_NOTES: list[str] = []
DEVIATION_NOTES: list[str] = [
    "Read-out architecture: single-query multi-head cross-attention (query is a learned "
    "[1,1,D] parameter; K/V linear from the input surface; SDPA; output LayerNorm), "
    "n_heads = the region's own encoder n_heads. Spec says 'small cross-attention "
    "read-out' and leaves the exact shape open; this is the simplest thing matching that "
    "description, and it is architecturally identical across arms (a)/(b)/(c) by "
    "construction, so 'identical params' is satisfied trivially rather than by matching "
    "two different designs' parameter counts after the fact.",
    "Training budget (600 steps, batch 256 text / 128 vl, AdamW lr 2e-3 with 60-step "
    "linear warmup then constant, weight decay 1e-4, grad-clip 1.0) is a from-scratch "
    "choice sized to 'small read-out over cached frozen activations, ~1h for 3 arms' "
    "(§4.1 cost table) on a lightly-loaded 3090 Ti; it is NOT taken from any existing "
    "receipt because no prior receipt trains a read-out of this shape. In-batch loss/acc "
    "at the final step is logged per arm so a reader can judge convergence directly "
    "rather than trusting the step count.",
    "Penultimate-block activations are the raw residual-stream output after "
    "blocks[:-2] finish (i.e. entering the last block), WITHOUT the model's final "
    "LayerNorm applied -- that norm's affine params were fit for the post-final-block "
    "distribution, not an earlier one, so applying it to an earlier block would not be "
    "a neutral choice. The read-out's own input projection can absorb the scale "
    "difference.",
    "'DEV half only' (§2.7.8) has no literal referent here -- §2.7.8 defines dev/sealed "
    "for the 5,120-item CROSS-FACULTY compose eval used by W5 onward, which does not "
    "exist yet and is not what W1d measures. Applied by principle instead: the read-out "
    "is fit ONLY on a slice of build_splits' train_pairs (the next 4,096 pairs after the "
    "512-item holdout, disjoint from it by construction), and the 512-item holdout -- "
    "already the production decision set W1 itself used -- is read exactly once, for "
    "the arm(a)-vs-arm(b) evaluation that produces the verdict.",
    "vl_latent's 'region's own task' is linear-probe top1/top5 classification (its "
    "receipt: 'objective: I-JEPA latent prediction; gated on linear probe, never on "
    "loss'), not retrieval. The read-out is therefore trained end-to-end with a joint "
    "linear classification head under cross-entropy on tiny-imagenet labels ('attentive "
    "probing'), which is the natural token-aware generalisation of the receipt's own "
    "linear probe over the pooled vector -- arm (b)'s broadcast-pool input makes this "
    "collapse back to exactly a linear probe over pool(), so the comparison is apples "
    "to apples with the production metric's own construction.",
    "vl_latent eval/train images are NOT the first N rows of their parquet in file order "
    "(unlike W1's own vl capture, and unlike this same script's text regions, where "
    "build_splits already shuffles). tiny-imagenet's parquets are class-grouped (500 "
    "contiguous rows/class train, 50/class valid, sorted ascending by label); the first "
    "512 valid rows span 11 of 200 classes and the first 4096 train rows span 9 -- "
    "confirmed by a direct parquet read before writing this fix, and unusable for a "
    "classification probe (any file-order prefix short of the whole split caps the "
    "read-out's ceiling on classes never seen, identically for every arm, which would "
    "understate everyone's accuracy without informing the a-vs-b comparison). Fix: "
    "stratified_decode() reads only the label column first (cheap), then decodes a "
    "seeded (VL_SHUFFLE_SEED), per-class-balanced sample of exactly N_HOLDOUT / "
    "N_TRAIN_POOL rows directly -- no on-disk cache of the full split, which does not "
    "fit this host's /tmp (~760MB free; the full 100k-row train split alone is 1.2GB "
    "uint8 -- hit mid-run and is why this function exists rather than a permute-then-"
    "slice of _decode_split's own full-split output). Rank and variance stats are "
    "computed on this stratified eval set, so they are not bit-identical to W1's README "
    "numbers for vl_latent specifically (text regions' rank numbers ARE bit-identical to "
    "W1, since build_splits' holdout order is untouched there -- confirmed in this run's "
    "own results against the W1 README table).",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
TextEncoder, TextEncoderConfig, info_nce = import_with_retry(
    "cogsyndelta.regions.text_encoder", ["TextEncoder", "TextEncoderConfig", "info_nce"]
)
from cogsyndelta.eval.benchmark import effective_rank as entropy_effective_rank  # noqa: E402
from cogsyndelta.eval.metrics import mean_reciprocal_rank, recall_at_k  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

IJEPA = JEPAConfig = None
_decode_split = _to_float = None


def shards(pattern: str, root: Path = CORPUS) -> list[str]:
    return sorted(str(p) for p in root.glob(pattern))


# Same three production text regions as W1 (docs/design/evidence/w1-token-rank-2026-09-02/
# measure_w1.py), copied verbatim so build_splits reproduces the IDENTICAL 512-item holdout
# W1 itself measured against. compress_repo_local is excluded: not production, not in scope.
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
    ),
}


# ---------------------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pr_effective_rank(x: torch.Tensor) -> float:
    """Participation-ratio effective rank: (sum s_i^2)^2 / sum s_i^4. Identical formula
    to measure_w1.py's pr_effective_rank, reproduced here so W1d never imports from a
    session-scratchpad script."""
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
    if x.size(0) < 2:
        return float("nan")
    return entropy_effective_rank(x, sample=x.size(0))


def within_between_variance(padded: torch.Tensor, mask: torch.Tensor | None) -> dict:
    """One-way decomposition on the probe's own token surface: within-item variance
    (avg squared distance of each valid token to ITS OWN item's mean) vs between-item
    variance (avg squared distance of each item's mean to the global mean of item means).
    'The tokens carry more than their mean' IS the statement that within-item variance is
    not ~0 relative to between-item variance -- rank does not measure this directly.
    """
    n = padded.size(0)
    item_means = []
    within_per_item = []
    for i in range(n):
        if mask is not None:
            m = mask[i].bool()
            toks = padded[i][m]
        else:
            toks = padded[i]
        if toks.size(0) == 0:
            continue
        mu = toks.mean(dim=0)
        item_means.append(mu)
        within_per_item.append(((toks - mu) ** 2).sum(dim=-1).mean().item())
    item_means_t = torch.stack(item_means)
    global_mu = item_means_t.mean(dim=0)
    between = ((item_means_t - global_mu) ** 2).sum(dim=-1).mean().item()
    within = float(sum(within_per_item) / len(within_per_item))
    return {
        "within_item_mean_sq_dist": within,
        "between_item_mean_sq_dist": between,
        "within_over_between": within / between if between > 0 else float("inf"),
        "pct_of_total_that_is_within": (
            100.0 * within / (within + between) if (within + between) > 0 else float("nan")
        ),
        "n_items": len(item_means),
    }


# ---------------------------------------------------------------------------------------
# Small matched cross-attention read-out. Identical class/dim/heads across arms (a)/(b)/(c)
# by construction, so 'identical parameter counts' never has to be verified after the fact.
# ---------------------------------------------------------------------------------------
class CrossAttnReadout(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        b, t, d = tokens.shape
        q = self.q_proj(self.query.expand(b, -1, -1))
        k, v = self.kv(tokens).chunk(2, dim=-1)
        q = q.view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if mask is not None:
            keep = mask.bool()
            empty = ~keep.any(dim=1)
            if empty.any():
                keep = keep.clone()
                keep[empty, 0] = True
            attn_mask = keep[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).reshape(b, d)
        return self.norm(self.out_proj(out))


def n_params(*modules: nn.Module) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())


def broadcast_pool(pooled: torch.Tensor, t: int) -> torch.Tensor:
    return pooled.unsqueeze(1).expand(-1, t, -1).contiguous()


def make_batch_order(n_items: int, steps: int, batch: int, seed: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, n_items, (batch,), generator=g) for _ in range(steps)]


def lr_at(step: int, lr: float, warmup: int) -> float:
    return lr * min(1.0, (step + 1) / max(1, warmup))


# ---------------------------------------------------------------------------------------
# Retrieval arm: text regions
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def eval_retrieval(readout, a_tok, a_mask, p_tok, p_mask, batch=64) -> dict:
    readout.eval()
    a_outs, p_outs = [], []
    for i in range(0, a_tok.size(0), batch):
        a_outs.append(
            readout(a_tok[i : i + batch].to(DEVICE), a_mask[i : i + batch].to(DEVICE)).float().cpu()
        )
        p_outs.append(
            readout(p_tok[i : i + batch].to(DEVICE), p_mask[i : i + batch].to(DEVICE)).float().cpu()
        )
    a = torch.cat(a_outs)
    p = torch.cat(p_outs)
    an = F.normalize(a, dim=-1)
    pn = F.normalize(p, dim=-1)
    scores = an @ pn.T
    relevant = torch.arange(a.size(0))
    return {
        "recall@1": recall_at_k(scores, relevant, 1),
        "recall@10": recall_at_k(scores, relevant, 10),
        "mrr": mean_reciprocal_rank(scores, relevant),
        "n_items": int(a.size(0)),
    }


def train_readout_retrieval(
    dim: int,
    n_heads: int,
    train_a_tok,
    train_a_mask,
    train_p_tok,
    train_p_mask,
    eval_a_tok,
    eval_a_mask,
    eval_p_tok,
    eval_p_mask,
    init_seed: int,
    batch_order: list[torch.Tensor],
) -> dict:
    torch.manual_seed(init_seed)
    readout = CrossAttnReadout(dim, n_heads).to(DEVICE)
    params = n_params(readout)
    untrained = eval_retrieval(readout, eval_a_tok, eval_a_mask, eval_p_tok, eval_p_mask)

    opt = torch.optim.AdamW(readout.parameters(), lr=READOUT_LR, weight_decay=READOUT_WEIGHT_DECAY)
    readout.train()
    last_stats = {}
    for step, idx in enumerate(batch_order):
        a_tok = train_a_tok[idx].to(DEVICE)
        a_mask = train_a_mask[idx].to(DEVICE)
        p_tok = train_p_tok[idx].to(DEVICE)
        p_mask = train_p_mask[idx].to(DEVICE)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, READOUT_LR, READOUT_WARMUP)
        a_out = readout(a_tok, a_mask)
        p_out = readout(p_tok, p_mask)
        loss, stats = info_nce(a_out, p_out, temperature=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(readout.parameters(), GRAD_CLIP)
        opt.step()
        last_stats = stats

    trained = eval_retrieval(readout, eval_a_tok, eval_a_mask, eval_p_tok, eval_p_mask)
    return {
        "n_params": params,
        "untrained": untrained,
        "trained": trained,
        "final_train_stats": last_stats,
    }


# ---------------------------------------------------------------------------------------
# Classification arm: vl_latent
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def eval_classify(readout, head, tok, labels, batch=64) -> dict:
    readout.eval()
    head.eval()
    logits_all = []
    for i in range(0, tok.size(0), batch):
        feat = readout(tok[i : i + batch].to(DEVICE), None)
        logits_all.append(head(feat).float().cpu())
    logits = torch.cat(logits_all)
    top1 = (logits.argmax(-1) == labels).float().mean().item()
    k = min(5, logits.size(1))
    top5 = (logits.topk(k, dim=-1).indices == labels.unsqueeze(-1)).any(-1).float().mean().item()
    return {"top1": top1, "top5": top5, "n_items": int(logits.size(0))}


def train_readout_classify(
    dim: int,
    n_heads: int,
    n_classes: int,
    train_tok,
    train_labels,
    eval_tok,
    eval_labels,
    init_seed: int,
    batch_order: list[torch.Tensor],
) -> dict:
    torch.manual_seed(init_seed)
    readout = CrossAttnReadout(dim, n_heads).to(DEVICE)
    head = nn.Linear(dim, n_classes).to(DEVICE)
    nn.init.zeros_(head.bias)
    params = n_params(readout, head)
    untrained = eval_classify(readout, head, eval_tok, eval_labels)

    opt = torch.optim.AdamW(
        list(readout.parameters()) + list(head.parameters()),
        lr=READOUT_LR,
        weight_decay=READOUT_WEIGHT_DECAY,
    )
    readout.train()
    head.train()
    last_loss = None
    for step, idx in enumerate(batch_order):
        tok = train_tok[idx].to(DEVICE)
        y = train_labels[idx].to(DEVICE)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, READOUT_LR, READOUT_WARMUP)
        feat = readout(tok, None)
        logits = head(feat)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(readout.parameters()) + list(head.parameters()), GRAD_CLIP)
        opt.step()
        last_loss = loss.item()

    trained = eval_classify(readout, head, eval_tok, eval_labels)
    return {
        "n_params": params,
        "untrained": untrained,
        "trained": trained,
        "final_train_loss": last_loss,
    }


# ---------------------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------------------
def apply_rule(a_pct: float, b_pct: float) -> tuple[str, float]:
    """Pre-committed, verbatim: arm(a) beats arm(b) by >= 2 points => OVERTURNED.
    Everything else (including arm(b) beating arm(a)) => CONFIRMED. This is a one-sided
    rule as written in the design doc -- it is not a symmetric |delta| test."""
    delta = a_pct - b_pct
    verdict = "OVERTURNED" if delta >= RULE_THRESHOLD_POINTS else "CONFIRMED"
    return verdict, delta


def verify_by_failing(a_pct: float, a_control_pct: float) -> dict:
    """arm(a) against arm(a), different read-out init seed, identical data. Simplest
    reading of 'assert it reports within 2 points' for a same-vs-same check: a SYMMETRIC
    |delta| < 2 test (there is no principled 'which one is (a)' between two runs of the
    same arm), rather than the one-sided production rule. Both readings are reported."""
    delta = a_pct - a_control_pct
    one_sided_verdict, _ = apply_rule(a_pct, a_control_pct)
    symmetric_ok = abs(delta) < RULE_THRESHOLD_POINTS
    return {
        "main_seed": SEED_MAIN,
        "control_seed": SEED_CONTROL,
        "main_pct": a_pct,
        "control_pct": a_control_pct,
        "delta": delta,
        "one_sided_rule_verdict_if_applied_literally": one_sided_verdict,
        "symmetric_within_2pt": symmetric_ok,
        "instrument_verdict": "CONFIRMED (gate can fail correctly)" if symmetric_ok else
        "GATE DID NOT SELF-CONFIRM -- seed-to-seed spread >= 2pp, instrument too noisy for a 2pp threshold",
    }


# ---------------------------------------------------------------------------------------
# Text region: capture final-block and penultimate-block token surfaces, bit-exact.
# ---------------------------------------------------------------------------------------
def load_text_model(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    enc_cfg = TextEncoderConfig(**dict(ckpt["config"]))
    model = TextEncoder(enc_cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sha = sha256_file(checkpoint_path)
    return model, enc_cfg, sha, ckpt.get("step")


@torch.no_grad()
def capture_text(model: "TextEncoder", ids: torch.Tensor, mask: torch.Tensor, batch: int = INFER_BATCH):
    """Manual replication of TextEncoder.forward, capturing final (post-norm) AND
    penultimate-block (pre-norm, raw residual) token surfaces alongside the pooled
    vector. No hooks: this literally IS the forward pass called explicitly, exactly as
    measure_w1.py's text_encode_from_ids does it, extended with a mid-loop capture."""
    model.eval()
    depth = len(model.blocks)
    finals, penults, pooled_chunks = [], [], []
    for i in range(0, ids.size(0), batch):
        b_ids = ids[i : i + batch]
        b_mask = mask[i : i + batch]
        b, t = b_ids.shape
        h = model.embed(b_ids) + model.pos_embed[:, :t]
        penult = None
        for bi, block in enumerate(model.blocks):
            h = block(h, b_mask)
            if bi == depth - 2:
                penult = h.clone()
        if penult is None:  # depth < 2, degenerate; fall back to the embedding
            penult = h.clone()
        h_final = model.norm(h)
        mask_f = b_mask.unsqueeze(-1).to(h_final.dtype)
        pooled = (h_final * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1e-6)
        pooled = model.proj(pooled)
        finals.append(h_final.float().cpu())
        penults.append(penult.float().cpu())
        pooled_chunks.append(pooled.float().cpu())
    return torch.cat(finals), torch.cat(penults), mask.cpu(), torch.cat(pooled_chunks)


def process_text_region(name: str, spec: dict, tok: Tokenizer) -> dict:
    t_region_start = time.time()
    log(f"=== {name}: build_splits (production config, seed={spec['seed']}) ===")
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
    holdout, train_pairs, meta = build_splits(cfg)
    log(f"    build_splits: {time.time()-t0:.1f}s, holdout={len(holdout)}, train_pairs={len(train_pairs)}")
    assert len(holdout) == N_HOLDOUT, f"expected {N_HOLDOUT} holdout pairs, got {len(holdout)}"
    if len(train_pairs) < N_TRAIN_POOL:
        raise ValueError(f"{name}: only {len(train_pairs)} train_pairs, need {N_TRAIN_POOL}")
    train_items = train_pairs[:N_TRAIN_POOL]
    VERIFIED_NOTES.append(
        f"{name}: train_items are train_pairs[:{N_TRAIN_POOL}] from build_splits' own "
        f"post-holdout, contamination-screened list -- disjoint from the {N_HOLDOUT}-item "
        f"holdout BY CONSTRUCTION (build_splits takes holdout = all_pairs[:holdout_pairs], "
        f"train_pairs = all_pairs[holdout_pairs:])."
    )

    log(f"=== {name}: load {spec['checkpoint']} ===")
    model, enc_cfg, sha, ckpt_step = load_text_model(spec["checkpoint"])
    max_len = enc_cfg.max_len

    eval_a = [a for a, _p in holdout]
    eval_p = [p for _a, p in holdout]
    train_a = [a for a, _p in train_items]
    train_p = [p for _a, p in train_items]

    eval_a_ids, eval_a_mask = _tokenize(tok, eval_a, max_len, DEVICE)
    eval_p_ids, eval_p_mask = _tokenize(tok, eval_p, max_len, DEVICE)
    train_a_ids, train_a_mask = _tokenize(tok, train_a, max_len, DEVICE)
    train_p_ids, train_p_mask = _tokenize(tok, train_p, max_len, DEVICE)

    # Correctness check: the manual capture must reproduce model(ids,mask) exactly.
    with torch.no_grad():
        eval_final_8, _, _, eval_pooled_8 = capture_text(model, eval_a_ids[:8], eval_a_mask[:8], batch=8)
        official_8 = model(eval_a_ids[:8], eval_a_mask[:8]).float().cpu()
    max_abs_diff = (eval_pooled_8 - official_8).abs().max().item()
    if max_abs_diff > 1e-4:
        UNRELIABLE_NOTES.append(
            f"{name}: manual capture diverged from model(ids,mask) by max_abs={max_abs_diff:.2e}"
        )
    else:
        VERIFIED_NOTES.append(
            f"{name}: manual forward capture (with mid-loop penultimate hook) matches "
            f"model(ids,mask) to max_abs={max_abs_diff:.2e} (8-item check)."
        )

    log(f"    capturing token surfaces (eval {N_HOLDOUT}, train {N_TRAIN_POOL})...")
    eval_a_final, eval_a_penult, eval_a_mask_c, eval_a_pooled = capture_text(model, eval_a_ids, eval_a_mask)
    eval_p_final, eval_p_penult, eval_p_mask_c, eval_p_pooled = capture_text(model, eval_p_ids, eval_p_mask)
    train_a_final, train_a_penult, train_a_mask_c, train_a_pooled = capture_text(model, train_a_ids, train_a_mask)
    train_p_final, train_p_penult, train_p_mask_c, train_p_pooled = capture_text(model, train_p_ids, train_p_mask)
    del model
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    dim = enc_cfg.dim
    n_heads = enc_cfg.n_heads

    # ---- rank + variance on the probe's own eval-set input activations ----
    eval_a_final_valid = torch.cat(
        [eval_a_final[i][eval_a_mask_c[i].bool()] for i in range(eval_a_final.size(0))], dim=0
    )
    eval_a_penult_valid = torch.cat(
        [eval_a_penult[i][eval_a_mask_c[i].bool()] for i in range(eval_a_penult.size(0))], dim=0
    )
    rank_report = {
        "final_block": {
            "pr_rank": pr_effective_rank(eval_a_final_valid),
            "entropy_rank": entropy_eff_rank_full(eval_a_final_valid),
            "n_tokens": int(eval_a_final_valid.size(0)),
        },
        "penultimate_block": {
            "pr_rank": pr_effective_rank(eval_a_penult_valid),
            "entropy_rank": entropy_eff_rank_full(eval_a_penult_valid),
            "n_tokens": int(eval_a_penult_valid.size(0)),
        },
        "pooled": {
            "pr_rank": pr_effective_rank(eval_a_pooled),
            "entropy_rank": entropy_eff_rank_full(eval_a_pooled),
        },
    }
    variance_report = {
        "final_block": within_between_variance(eval_a_final, eval_a_mask_c),
        "penultimate_block": within_between_variance(eval_a_penult, eval_a_mask_c),
    }

    # ---- shared batch order: identical items, identical order, for arms (a)/(b)/(c) ----
    batch_order = make_batch_order(N_TRAIN_POOL, READOUT_STEPS, READOUT_BATCH_TEXT, DATA_ORDER_SEED)

    train_a_pool_bcast = broadcast_pool(train_a_pooled, train_a_final.size(1))
    train_p_pool_bcast = broadcast_pool(train_p_pooled, train_p_final.size(1))
    eval_a_pool_bcast = broadcast_pool(eval_a_pooled, eval_a_final.size(1))
    eval_p_pool_bcast = broadcast_pool(eval_p_pooled, eval_p_final.size(1))

    log(f"    training arm (a) final-block tokens, seed={SEED_MAIN}...")
    arm_a = train_readout_retrieval(
        dim, n_heads,
        train_a_final, train_a_mask_c, train_p_final, train_p_mask_c,
        eval_a_final, eval_a_mask_c, eval_p_final, eval_p_mask_c,
        SEED_MAIN, batch_order,
    )
    log(f"    training arm (b) pool-broadcast, seed={SEED_MAIN}...")
    arm_b = train_readout_retrieval(
        dim, n_heads,
        train_a_pool_bcast, train_a_mask_c, train_p_pool_bcast, train_p_mask_c,
        eval_a_pool_bcast, eval_a_mask_c, eval_p_pool_bcast, eval_p_mask_c,
        SEED_MAIN, batch_order,
    )
    log(f"    training arm (c) penultimate-block tokens, seed={SEED_MAIN}...")
    arm_c = train_readout_retrieval(
        dim, n_heads,
        train_a_penult, train_a_mask_c, train_p_penult, train_p_mask_c,
        eval_a_penult, eval_a_mask_c, eval_p_penult, eval_p_mask_c,
        SEED_MAIN, batch_order,
    )
    log(f"    verify-by-failing: arm (a) again, seed={SEED_CONTROL}...")
    arm_a_control = train_readout_retrieval(
        dim, n_heads,
        train_a_final, train_a_mask_c, train_p_final, train_p_mask_c,
        eval_a_final, eval_a_mask_c, eval_p_final, eval_p_mask_c,
        SEED_CONTROL, batch_order,
    )

    a_pct = arm_a["trained"]["recall@1"] * 100
    b_pct = arm_b["trained"]["recall@1"] * 100
    c_pct = arm_c["trained"]["recall@1"] * 100
    a_control_pct = arm_a_control["trained"]["recall@1"] * 100

    verdict, delta_ab = apply_rule(a_pct, b_pct)
    verify = verify_by_failing(a_pct, a_control_pct)
    penult_delta = c_pct - a_pct

    anomaly_flags = []
    if delta_ab <= -RULE_THRESHOLD_POINTS:
        anomaly_flags.append(
            f"arm(b) beat arm(a) by {-delta_ab:.2f}pp -- attending over a broadcast pool "
            f"outperformed attending over real tokens by more than the decision threshold; "
            f"this is not predicted by the mechanism and should be treated as a possible "
            f"read-out optimisation artefact on arm(a), not evidence tokens carry negative "
            f"information."
        )

    result = {
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": sha,
        "checkpoint_step": ckpt_step,
        "dim": dim,
        "n_heads": n_heads,
        "max_len": max_len,
        "n_holdout": N_HOLDOUT,
        "n_train_pool": N_TRAIN_POOL,
        "metric": "recall@1 (region receipt metric), percentage points",
        "arms": {
            "a_final_tokens": arm_a,
            "b_pool_broadcast": arm_b,
            "c_penultimate_tokens": arm_c,
            "a_control_seed": arm_a_control,
        },
        "arm_a_pct": a_pct,
        "arm_b_pct": b_pct,
        "arm_c_pct": c_pct,
        "delta_a_minus_b_points": delta_ab,
        "rule_threshold_points": RULE_THRESHOLD_POINTS,
        "verdict": verdict,
        "verify_by_failing": verify,
        "penultimate_arm": {
            "recall_at_1_pct": c_pct,
            "delta_vs_final_block_points": penult_delta,
            "interpretation": (
                "penultimate-block read-out within noise of final-block -- L_token could be "
                "attached at either block" if abs(penult_delta) < RULE_THRESHOLD_POINTS else
                ("penultimate-block read-out MEANINGFULLY WORSE than final-block -- keep "
                 "L_token at the final block (ADOPTED option), penultimate fallback not "
                 "indicated" if penult_delta < 0 else
                 "penultimate-block read-out MEANINGFULLY BETTER than final-block -- the "
                 "penultimate fallback (§4.0 option 2) is evidenced, not just available")
            ),
        },
        "anomaly_flags": anomaly_flags,
        "rank_on_probe_input_activations": rank_report,
        "within_between_token_variance": variance_report,
        "wall_time_s": time.time() - t_region_start,
    }
    log(f"=== {name}: DONE in {result['wall_time_s']:.1f}s -- verdict {verdict} "
        f"(a={a_pct:.2f}%, b={b_pct:.2f}%, delta={delta_ab:+.2f}pp) ===")
    return result


# ---------------------------------------------------------------------------------------
# vl_latent (optional, included if checkpoint + eval images are present)
# ---------------------------------------------------------------------------------------
def try_load_vl():
    global IJEPA, JEPAConfig, _decode_split, _to_float
    ckpt_path = Path("/akula-data/csd/receipts/vl_latent-checkpoints/step-8000.pt")
    valid_shards = shards("vl/tiny-imagenet/data/valid-*.parquet", root=CORPUS)
    train_shards = shards("vl/tiny-imagenet/data/train-*.parquet", root=CORPUS)
    if not ckpt_path.is_file() or not valid_shards or not train_shards:
        return None, None, None
    (IJEPA, JEPAConfig) = import_with_retry("cogsyndelta.model.vl_jepa", ["IJEPA", "JEPAConfig"])
    (_decode_split, _to_float) = import_with_retry(
        "cogsyndelta.regions.vl_pretrain", ["_decode_split", "_to_float"]
    )
    return ckpt_path, valid_shards, train_shards


def stratified_decode(shards_list: list[str], image_col: str, label_col: str, size: int, n_total: int, seed: int):
    """Decode n_total images STRATIFIED BY LABEL, with no on-disk cache.

    tiny-imagenet's parquet is class-grouped in file order (500 contiguous rows/class
    train, 50/class valid; confirmed by direct read), so neither a file-order prefix nor
    a shuffle-then-slice of a bounded prefix is guaranteed to cover all classes -- and
    _decode_split's own on-disk .npy cache for the FULL split does not fit: the full
    100,000-image train split is 1.2 GB uint8, and this host's /tmp partition had ~760 MB
    free when this was written. This function reads only the label column first (cheap),
    picks a seeded-random, per-class-balanced set of row indices, and decodes ONLY those
    rows -- the per-row PIL decode mirrors vl_pretrain._decode_split exactly, but nothing
    is written to disk.
    """
    import io as _io

    import numpy as _np
    import pyarrow.parquet as pq
    from PIL import Image

    by_class: dict[int, list[tuple[str, int]]] = {}
    for path in shards_list:
        t = pq.read_table(path, columns=[label_col])
        labs = t.column(label_col).to_pylist()
        for i, lab in enumerate(labs):
            by_class.setdefault(int(lab), []).append((path, i))

    classes = sorted(by_class)
    per_class = max(1, -(-n_total // len(classes)))  # ceil
    g = torch.Generator().manual_seed(seed)
    picked: list[tuple[str, int, int]] = []
    for lab in classes:
        items = by_class[lab]
        k = min(per_class, len(items))
        perm = torch.randperm(len(items), generator=g)[:k].tolist()
        picked.extend((items[p][0], items[p][1], lab) for p in perm)
    order = torch.randperm(len(picked), generator=g)[: min(n_total, len(picked))].tolist()
    picked = [picked[o] for o in order]

    by_path: dict[str, list[tuple[int, int]]] = {}
    for pos, (path, i, _lab) in enumerate(picked):
        by_path.setdefault(path, []).append((i, pos))

    images: list = [None] * len(picked)
    labels = [0] * len(picked)
    for path, idx_pos in by_path.items():
        pos_by_row = dict(idx_pos)
        t = pq.read_table(path, columns=[image_col, label_col])
        col_img = t.column(image_col)
        col_lab = t.column(label_col)
        for i, pos in pos_by_row.items():
            rec = col_img[i].as_py()
            raw = rec["bytes"] if isinstance(rec, dict) else rec
            with Image.open(_io.BytesIO(raw)) as handle:
                rgb = handle.convert("RGB")
                if rgb.size != (size, size):
                    rgb = rgb.resize((size, size), Image.Resampling.BICUBIC)
                arr = _np.asarray(rgb, dtype=_np.uint8).transpose(2, 0, 1)
            images[pos] = arr
            labels[pos] = int(col_lab[i].as_py())
    x = torch.from_numpy(_np.stack(images))
    y = torch.tensor(labels, dtype=torch.int64)
    return x, y


@torch.no_grad()
def capture_vl(encoder, images_f: torch.Tensor, batch: int = INFER_BATCH):
    encoder.eval()
    depth = len(encoder.blocks)
    finals, penults, pooled_chunks = [], [], []
    for i in range(0, images_f.size(0), batch):
        x = images_f[i : i + batch]
        h = encoder.patch_embed(x) + encoder.pos_embed
        penult = None
        for bi, block in enumerate(encoder.blocks):
            h = block(h)
            if bi == depth - 2:
                penult = h.clone()
        if penult is None:
            penult = h.clone()
        h_final = encoder.norm(h)
        pooled = h_final.mean(dim=1)
        finals.append(h_final.float().cpu())
        penults.append(penult.float().cpu())
        pooled_chunks.append(pooled.float().cpu())
    return torch.cat(finals), torch.cat(penults), torch.cat(pooled_chunks)


def process_vl_region(ckpt_path: Path, valid_shards: list[str], train_shards: list[str]) -> dict:
    t_region_start = time.time()
    log("=== vl_latent: stratified decode of valid + train (no on-disk cache) ===")
    x_eval, y_eval = stratified_decode(valid_shards, "image", "label", 64, N_HOLDOUT, VL_SHUFFLE_SEED)
    x_train, y_train = stratified_decode(train_shards, "image", "label", 64, N_TRAIN_POOL, VL_SHUFFLE_SEED + 1)

    assert x_eval.size(0) == N_HOLDOUT, f"expected {N_HOLDOUT} eval images, got {x_eval.size(0)}"
    assert x_train.size(0) == N_TRAIN_POOL, f"expected {N_TRAIN_POOL} train images, got {x_train.size(0)}"
    n_classes_eval = len(set(y_eval.tolist()))
    n_classes_train = len(set(y_train.tolist()))
    n_classes = max(n_classes_eval, n_classes_train, int(y_eval.max().item()) + 1, int(y_train.max().item()) + 1)
    log(f"    eval={x_eval.size(0)} ({n_classes_eval} classes), "
        f"train={x_train.size(0)} ({n_classes_train} classes)")
    VERIFIED_NOTES.append(
        f"vl_latent: eval images are a seeded, class-stratified {N_HOLDOUT}-sample of the "
        f"valid split ({n_classes_eval} of 200 classes present), train images a seeded, "
        f"class-stratified {N_TRAIN_POOL}-sample of the train split ({n_classes_train} of "
        f"200 classes present) -- disjoint by construction (different source file, valid "
        f"vs train), not by slicing. No on-disk decode cache was used (see deviation_notes "
        f"for why: the full-split cache route does not fit this host's /tmp partition)."
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    jcfg = JEPAConfig(**dict(ckpt["config"]))
    model = IJEPA(jcfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sha = sha256_file(str(ckpt_path))

    images_eval = _to_float(x_eval, DEVICE)
    images_train = _to_float(x_train, DEVICE)

    with torch.no_grad():
        eval_final_8, _, eval_pooled_8 = capture_vl(model.target_encoder, images_eval[:8], batch=8)
        official_8 = model.encode(images_eval[:8]).float().cpu()
    max_abs_diff = (eval_pooled_8 - official_8).abs().max().item()
    if max_abs_diff > 1e-4:
        UNRELIABLE_NOTES.append(f"vl_latent: manual capture diverged by max_abs={max_abs_diff:.2e}")
    else:
        VERIFIED_NOTES.append(
            f"vl_latent: manual capture (with mid-loop penultimate hook) matches "
            f"IJEPA.encode() to max_abs={max_abs_diff:.2e} (8-item check)."
        )

    log(f"    capturing patch-token surfaces (eval {N_HOLDOUT}, train {N_TRAIN_POOL})...")
    eval_final, eval_penult, eval_pooled = capture_vl(model.target_encoder, images_eval)
    train_final, train_penult, train_pooled = capture_vl(model.target_encoder, images_train)
    del model
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    dim = jcfg.dim
    n_heads = jcfg.n_heads

    eval_final_valid = eval_final.reshape(-1, dim)
    eval_penult_valid = eval_penult.reshape(-1, dim)
    rank_report = {
        "final_block": {
            "pr_rank": pr_effective_rank(eval_final_valid),
            "entropy_rank": entropy_eff_rank_full(eval_final_valid),
            "n_tokens": int(eval_final_valid.size(0)),
        },
        "penultimate_block": {
            "pr_rank": pr_effective_rank(eval_penult_valid),
            "entropy_rank": entropy_eff_rank_full(eval_penult_valid),
            "n_tokens": int(eval_penult_valid.size(0)),
        },
        "pooled": {
            "pr_rank": pr_effective_rank(eval_pooled),
            "entropy_rank": entropy_eff_rank_full(eval_pooled),
        },
    }
    variance_report = {
        "final_block": within_between_variance(eval_final, None),
        "penultimate_block": within_between_variance(eval_penult, None),
    }

    batch_order = make_batch_order(N_TRAIN_POOL, READOUT_STEPS, READOUT_BATCH_VL, DATA_ORDER_SEED)
    train_pool_bcast = broadcast_pool(train_pooled, train_final.size(1))
    eval_pool_bcast = broadcast_pool(eval_pooled, eval_final.size(1))

    log(f"    training arm (a) final-block patch tokens, seed={SEED_MAIN}...")
    arm_a = train_readout_classify(
        dim, n_heads, n_classes, train_final, y_train, eval_final, y_eval, SEED_MAIN, batch_order
    )
    log(f"    training arm (b) pool-broadcast, seed={SEED_MAIN}...")
    arm_b = train_readout_classify(
        dim, n_heads, n_classes, train_pool_bcast, y_train, eval_pool_bcast, y_eval, SEED_MAIN, batch_order
    )
    log(f"    training arm (c) penultimate-block patch tokens, seed={SEED_MAIN}...")
    arm_c = train_readout_classify(
        dim, n_heads, n_classes, train_penult, y_train, eval_penult, y_eval, SEED_MAIN, batch_order
    )
    log(f"    verify-by-failing: arm (a) again, seed={SEED_CONTROL}...")
    arm_a_control = train_readout_classify(
        dim, n_heads, n_classes, train_final, y_train, eval_final, y_eval, SEED_CONTROL, batch_order
    )

    a_pct = arm_a["trained"]["top1"] * 100
    b_pct = arm_b["trained"]["top1"] * 100
    c_pct = arm_c["trained"]["top1"] * 100
    a_control_pct = arm_a_control["trained"]["top1"] * 100

    verdict, delta_ab = apply_rule(a_pct, b_pct)
    verify = verify_by_failing(a_pct, a_control_pct)
    penult_delta = c_pct - a_pct

    anomaly_flags = []
    if delta_ab <= -RULE_THRESHOLD_POINTS:
        anomaly_flags.append(
            f"arm(b) beat arm(a) by {-delta_ab:.2f}pp -- see the text-region note; treated "
            f"as a possible optimisation artefact, not evidence against tokens."
        )

    result = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha,
        "checkpoint_step": ckpt.get("step"),
        "dim": dim,
        "n_heads": n_heads,
        "n_patches": jcfg.n_patches,
        "n_classes": n_classes,
        "n_holdout": N_HOLDOUT,
        "n_train_pool": N_TRAIN_POOL,
        "metric": "top1 (region receipt metric: linear-probe classification), percentage points",
        "arms": {
            "a_final_tokens": arm_a,
            "b_pool_broadcast": arm_b,
            "c_penultimate_tokens": arm_c,
            "a_control_seed": arm_a_control,
        },
        "arm_a_pct": a_pct,
        "arm_b_pct": b_pct,
        "arm_c_pct": c_pct,
        "delta_a_minus_b_points": delta_ab,
        "rule_threshold_points": RULE_THRESHOLD_POINTS,
        "verdict": verdict,
        "verify_by_failing": verify,
        "penultimate_arm": {
            "top1_pct": c_pct,
            "delta_vs_final_block_points": penult_delta,
        },
        "anomaly_flags": anomaly_flags,
        "rank_on_probe_input_activations": rank_report,
        "within_between_token_variance": variance_report,
        "wall_time_s": time.time() - t_region_start,
    }
    log(f"=== vl_latent: DONE in {result['wall_time_s']:.1f}s -- verdict {verdict} "
        f"(a={a_pct:.2f}%, b={b_pct:.2f}%, delta={delta_ab:+.2f}pp) ===")
    return result


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    results: dict = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "regions": {},
    }

    tok = Tokenizer.from_file(TOKENIZER_PATH)
    VERIFIED_NOTES.append(f"tokenizer loaded from {TOKENIZER_PATH}, vocab_size={tok.get_vocab_size()}")

    for name, spec in TEXT_REGION_SPECS.items():
        try:
            results["regions"][name] = process_text_region(name, spec, tok)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            UNRELIABLE_NOTES.append(f"{name}: FAILED -- {e!r}\n{tb}")
            log(f"!!! {name} failed: {e!r}")

    try:
        ckpt_path, valid_shards, train_shards = try_load_vl()
        if ckpt_path is None:
            INFERRED_NOTES.append("vl_latent: checkpoint or eval/train images not found; skipped.")
        else:
            results["regions"]["vl_latent"] = process_vl_region(ckpt_path, valid_shards, train_shards)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        UNRELIABLE_NOTES.append(f"vl_latent: FAILED -- {e!r}\n{tb}")
        log(f"!!! vl_latent failed: {e!r}")

    results["verified_notes"] = VERIFIED_NOTES
    results["inferred_notes"] = INFERRED_NOTES
    results["unreliable_notes"] = UNRELIABLE_NOTES
    results["deviation_notes"] = DEVIATION_NOTES
    results["wall_time_s_total"] = time.time() - t_start
    results["config"] = {
        "readout_steps": READOUT_STEPS,
        "readout_batch_text": READOUT_BATCH_TEXT,
        "readout_batch_vl": READOUT_BATCH_VL,
        "readout_lr": READOUT_LR,
        "readout_warmup": READOUT_WARMUP,
        "seed_main": SEED_MAIN,
        "seed_control": SEED_CONTROL,
        "data_order_seed": DATA_ORDER_SEED,
        "rule_threshold_points": RULE_THRESHOLD_POINTS,
        "n_holdout": N_HOLDOUT,
        "n_train_pool": N_TRAIN_POOL,
    }

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote {OUT_DIR / 'results.json'} -- total wall time {results['wall_time_s_total']:.1f}s")
    return results


if __name__ == "__main__":
    main()
