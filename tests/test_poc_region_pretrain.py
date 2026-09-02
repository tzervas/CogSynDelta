import pytest
import torch
from cogsyndelta.poc.train import train_latent_vae_on_public_split

@pytest.mark.cpu
def test_latent_vae_pretrain_loss_decreases_on_wikitext2_train():
    torch.manual_seed(42)
    result = train_latent_vae_on_public_split(steps=20, batch_size=8, input_dim=768)
    assert result['last_loss'] < result['first_loss']
    assert result['split'] == 'train'