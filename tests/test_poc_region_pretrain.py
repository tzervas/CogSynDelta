import pytest
import torch
from cogsyndelta.poc.train import train_latent_vae_on_public_split

@pytest.mark.cpu
@pytest.mark.seed(42)
def test_latent_vae_pretrain_loss_decreases_on_wikitext2_train():
    result = train_latent_vae_on_public_split(
        dataset='Salesforce/wikitext',
        config='wikitext-2-raw-v1',
        split='train',
        steps=20,
        batch_size=8,
        input_dim=768  # Adjust this to match your PoC
    )
    assert result['last_loss'] < result['first_loss']
    assert result['split'] == 'train'