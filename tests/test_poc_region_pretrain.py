import pytest
import torch
from cogsyndelta.poc.train import train_latent_vae_on_public_split

def test_latent_vae_pretrain_loss_decreases_on_wikitext2_train():
    torch.manual_seed(42)
    config = {
        'input_dim': 768,  # Adjust this based on your PoC
        'batch_size': 8,
        'steps': 20,
        'split': 'train'
    }
    result = train_latent_vae_on_public_split(config)
    assert result['last_loss'] < result['first_loss']
    assert result['split'] == 'train'