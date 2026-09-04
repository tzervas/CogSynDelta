"""B3: `cogsyndelta.eval.benchmark.pr_effective_rank` must reproduce
``docs/design/evidence/w1-token-rank-2026-09-02/measure_w1.py``'s own `pr_effective_rank`
-- the function the W1/W1d/W4 retrain-gate rank clauses were pre-committed against --
exactly, not merely in shape.

The frozen evidence script is never imported or executed here: it inserts the MAIN
checkout's `src/` onto `sys.path` at module scope (`REPO = "/home/kang/.../CogSynDelta"`)
and its README states "Nothing here is regenerated on read" -- running it, even by a
plain `import`, would both violate that and risk shadowing this worktree's own
`cogsyndelta` package for the rest of the test session. Instead this test extracts ONLY
the `pr_effective_rank` function's source via `ast` and executes that fragment in an
isolated namespace, so the comparison is against the frozen script's actual code, not a
hand-copied restatement of it that could drift unnoticed.

Only summary numbers (`results.json`) are checked into the evidence directory, not the
raw per-item token/pooled arrays the original run computed them from, so the comparison
below runs on small synthetic surfaces (a rank-one matrix, isotropic noise, degenerate
<2-row inputs) rather than re-deriving W1's exact production numbers.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch

from cogsyndelta.eval.benchmark import participation_ratio, pr_effective_rank

# `_final_block_rank_stats` (via `cogsyndelta.regions.pretrain`) pulls in the `train`
# dependency group (`tokenizers`, transitively `pyarrow`) at import time; skip the whole
# module rather than error at collection where that group is not installed (e.g.
# `scripts/ci_local.sh`'s own minimal dev-group venv) -- matching
# `tests/test_token_aware_objective.py`'s own convention.
pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from cogsyndelta.regions.pretrain import _final_block_rank_stats, _tokenize
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

pytestmark = pytest.mark.cpu

_W1_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "design"
    / "evidence"
    / "w1-token-rank-2026-09-02"
    / "measure_w1.py"
)


def _load_w1_pr_effective_rank():
    """Extract just the `pr_effective_rank` function body from the frozen W1 evidence
    script via `ast`, and exec it in a bare namespace -- never a plain `import` of the
    module (see module docstring)."""
    source = _W1_SCRIPT.read_text()
    tree = ast.parse(source, filename=str(_W1_SCRIPT))
    fn_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "pr_effective_rank"
    )
    fn_source = ast.get_source_segment(source, fn_node)
    assert fn_source is not None, "could not extract pr_effective_rank source from measure_w1.py"
    # The function body references the module-level `DEVICE` constant (measure_w1.py's
    # own `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`);
    # supply the identical value rather than re-deriving a different one.
    namespace: dict[str, object] = {
        "torch": torch,
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    exec(fn_source, namespace)  # noqa: S102 -- extracted, reviewed source, not user input
    return namespace["pr_effective_rank"]


def test_w1_evidence_script_still_defines_pr_effective_rank() -> None:
    """Fails loudly if the frozen script is ever edited or moved, instead of the AST
    extraction above silently finding nothing and every comparison test skipping."""
    w1_fn = _load_w1_pr_effective_rank()
    assert callable(w1_fn)


@pytest.mark.parametrize(
    "shape",
    [(0, 4), (1, 4), (1, 1), (2, 4), (2, 1), (50, 16), (600, 32)],
)
def test_pr_effective_rank_reproduces_measure_w1_to_1e6(shape: tuple[int, int]) -> None:
    w1_pr_effective_rank = _load_w1_pr_effective_rank()
    torch.manual_seed(0)
    x = torch.randn(*shape)

    src_value = pr_effective_rank(x)
    w1_value = w1_pr_effective_rank(x)

    if math.isnan(w1_value) or math.isnan(src_value):
        assert math.isnan(w1_value) and math.isnan(src_value), (
            f"nan disagreement at shape={shape}: src={src_value} w1={w1_value}"
        )
    else:
        assert src_value == pytest.approx(w1_value, abs=1e-6), (
            f"shape={shape}: src={src_value} w1={w1_value}"
        )


def test_pr_effective_rank_of_a_rank_one_matrix_is_one_and_matches_w1() -> None:
    w1_pr_effective_rank = _load_w1_pr_effective_rank()
    direction = torch.tensor([1.0, 2.0, -1.0, 0.5])
    x = torch.outer(torch.linspace(1.0, 5.0, 20), direction)
    assert pr_effective_rank(x) == pytest.approx(1.0, abs=1e-4)
    assert pr_effective_rank(x) == pytest.approx(w1_pr_effective_rank(x), abs=1e-6)


def test_pr_effective_rank_below_two_rows_is_nan_not_zero() -> None:
    """The exact behaviour `participation_ratio` (0.0 below 2 rows) does NOT share --
    that mismatch is why the two functions are not interchangeable for the W1-lineage
    gates; see `pr_effective_rank`'s docstring."""
    assert math.isnan(pr_effective_rank(torch.zeros(1, 4)))
    assert math.isnan(pr_effective_rank(torch.zeros(0, 4)))
    # participation_ratio, by contrast, returns a clean 0.0 in the same case.
    assert participation_ratio(torch.zeros(1, 4)) == 0.0
    assert participation_ratio(torch.zeros(0, 4)) == 0.0


def test_pr_effective_rank_never_subsamples_unlike_participation_ratio() -> None:
    """`participation_ratio`'s default `sample=2048` would subsample a 3000-row surface;
    `pr_effective_rank` must run the exact SVD on all 3000 rows every time -- the same
    call twice must be identical (no seeded-but-still-random subsampling in play)."""
    torch.manual_seed(0)
    x = torch.randn(3000, 8)
    first = pr_effective_rank(x)
    second = pr_effective_rank(x)
    assert first == second


def test_final_block_rank_stats_measures_only_the_first_pair_side() -> None:
    """B3: `_final_block_rank_stats` (`regions/pretrain.py`) must measure the SAME side
    of the held-out pairs W1's pre-committed harness did -- `anchors = [a for a, _p in
    holdout]` in `measure_w1.py` -- not anchors and positives concatenated. `n_tokens`
    (and therefore every rank figure derived from the token-global surface) has to equal
    the token count of the FIRST side alone. Deliberately asymmetric pair lengths (short
    anchor, long positive) so a bug that pooled both sides in would move `n_tokens` by a
    detectable amount, not by a coincidental match."""
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = Whitespace()
    vocab = [f"w{i}" for i in range(60)]
    texts_for_training = [" ".join(vocab)]
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(texts_for_training, trainer=trainer)
    # `tok.get_vocab_size()`, not a hardcoded guess: the trainer's special tokens push the
    # real vocab past len(vocab) alone, and an undersized encoder vocab raises IndexError
    # in `F.embedding` the moment a word past its size is tokenised.
    cfg = TextEncoderConfig(vocab_size=tok.get_vocab_size(), dim=16, depth=1, n_heads=2, max_len=32)
    model = TextEncoder(cfg)

    pairs = [
        ("w0 w1 w2", "w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14"),
        ("w15 w16", "w17 w18 w19 w20 w21 w22 w23 w24 w25 w26"),
    ]
    device = torch.device("cpu")
    stats = _final_block_rank_stats(model, tok, pairs, max_len=32, device=device)

    anchor_ids, anchor_mask = _tokenize(tok, [a for a, _b in pairs], 32, device)
    _h, tmask = model.tokens(anchor_ids, anchor_mask)
    expected_n_tokens = float(tmask.bool().sum().item())

    both_ids, both_mask = _tokenize(
        tok, [a for a, _b in pairs] + [b for _a, b in pairs], 32, device
    )
    _h2, tmask2 = model.tokens(both_ids, both_mask)
    both_sides_n_tokens = float(tmask2.bool().sum().item())

    assert stats["n_tokens"] == pytest.approx(expected_n_tokens)
    # Guard the guard: the two token counts must actually differ for this fixture, or the
    # assertion above would pass by coincidence rather than by measuring the right side.
    assert expected_n_tokens != both_sides_n_tokens
    assert stats["n_tokens"] != pytest.approx(both_sides_n_tokens)
