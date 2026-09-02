import pytest
from cogsyndelta.poc.train import train_latent_vae_on_public_split

def test_latent_vae_pretrain_loss_decreases_on_wikitext2_train():
    result = train_latent_vae_on_public_split(
        seed=42,
        steps=20,
        batch_size=8,
        input_dim=768  # Adjust this based on the actual input_dim of your PoC
    )
    assert result['last_loss'] < result['first_loss'], "Loss did not decrease"
    assert result['split'] == 'train', "Split is not 'train'"
