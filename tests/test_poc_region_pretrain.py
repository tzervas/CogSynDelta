import pytest
import torch

from cogsyndelta.poc.train import train_latent_vae_on_public_split


@pytest.mark.cpu
def test_latent_vae_pretrain_loss_decreases_on_wikitext2_train():
    torch.manual_seed(42)
    result = train_latent_vae_on_public_split(
        dataset="Salesforce/wikitext",
        config="wikitext-2-raw-v1",
        split="train",
        steps=20,
        batch_size=8,
        input_dim=768,
    )
    assert result["last_loss"] < result["first_loss"]
    assert result["split"] == "train"


@pytest.mark.cpu
def test_encoder_preserves_lexical_similarity():
    """Related lines must encode more similarly than unrelated ones.

    This is the property the previous encoder lacked, and its absence was invisible:
    it hashed the WHOLE line with blake2b and expanded the digest. A cryptographic hash
    is designed to decorrelate on any input change, so near-identical sentences produced
    orthogonal vectors. The pretrain loss still fell — the VAE was memorising fixed
    pseudorandom points — so no existing assertion could catch it.

    Without this test, swapping the encoder back for anything that destroys similarity
    would look green again. Measured separation at time of writing: ~0.79.
    """
    from cogsyndelta.poc.train import encode_public_line

    related = [
        ("The cat sat on the mat.", "The cat sat on a mat."),
        (
            "Robert Boulter is an English film actor.",
            "Robert Boulter is an English television actor.",
        ),
    ]
    unrelated = [
        ("The cat sat on the mat.", "Quantum chromodynamics describes the strong interaction."),
        (
            "Robert Boulter is an English film actor.",
            "Sedimentary rock forms from compressed particles.",
        ),
    ]

    def sim(a: str, b: str) -> float:
        va, vb = encode_public_line(a, 768), encode_public_line(b, 768)
        return torch.nn.functional.cosine_similarity(va, vb, dim=0).item()

    related_mean = sum(sim(a, b) for a, b in related) / len(related)
    unrelated_mean = sum(sim(a, b) for a, b in unrelated) / len(unrelated)

    assert related_mean > 0.5, f"related lines encode too far apart: {related_mean:.3f}"
    assert related_mean - unrelated_mean > 0.3, (
        f"encoder carries no lexical signal: related={related_mean:.3f} "
        f"unrelated={unrelated_mean:.3f}"
    )


@pytest.mark.cpu
def test_encoder_is_deterministic_across_calls():
    """Same text must encode identically, or checkpoints are unreproducible.

    Guards against a future switch to Python's builtin hash(), which is salted per
    process unless PYTHONHASHSEED is pinned.
    """
    from cogsyndelta.poc.train import encode_public_line

    text = "Valkyria Chronicles is a tactical role-playing video game."
    assert torch.equal(encode_public_line(text, 256), encode_public_line(text, 256))
