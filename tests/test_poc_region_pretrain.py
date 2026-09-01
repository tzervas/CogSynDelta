import pytest
from cogsyndelta.poc.train import train_latent_vae_on_public_split

def test_latent_vae_pretrain_loss_decreases_on_wikitext2_train():
    seed = 42
    steps = 20
    batch_size = 8
    input_dim = 768  # Example input dimension, adjust as needed
    result = train_latent_vae_on_public_split(seed, steps, batch_size, input_dim)
    assert result['last_loss'] < result['first_loss']
    assert result['split'] == 'train'