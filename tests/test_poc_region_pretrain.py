import pytest
import torch
from cogsyndelta.poc.train import train_latent_vae_on_public_split

@pytest.mark.cpu
def test_latent_vae_pretrain_loss_decreases_on_wikitext2_train():
    torch.manual_seed(42)
    config = {
        'input_dim': 1024,  # Match PoC input_dim
        'batch_size': 8,
        'steps': 20,
        'split': 'train'
    }
    result = train_latent_vae_on_public_split(config)
    assert result['last_loss'] < result['first_loss']
    assert result['split'] == 'train'