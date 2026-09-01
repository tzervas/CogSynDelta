import pytest
import torch
from cogsyndelta.poc.train import train_latent_vae_on_public_split

def test_latent_vae_pretrain_loss_decreases_on_wikitext2_train():
    torch.manual_seed(42)
    result = train_latent_vae_on_public_split(
        dataset='Salesforce/wikitext',
        config='wikitext-2-raw-v1',
        split='train',
        steps=20,
        batch_size=8,
        input_dim=768
    )
    assert result['last_loss'] < result['first_loss'], 'Loss did not decrease'
    assert result['split'] == 'train', 'Incorrect split'
