import pytest
import torch
from cogsyndelta.poc.train import train_latent_vae_on_public_split

def test_latent_vae_pretrain_loss_decreases_on_wikitext2_train():
    torch.manual_seed(42)
    config = {
        'input_dim': 1024,  # Example input dimension, adjust as needed
        'batch_size': 8,
        'steps': 20,
        'split': 'train'
    }
    result = train_latent_vae_on_public_split(config)
    assert result['last_loss'] < result['first_loss'], 'Loss did not decrease'
    assert result['split'] == 'train', 'Incorrect split'
