import pytest
import torch
import cogsyndelta.poc.train.train_latent_vae_on_public_split

def test_latent_vae_pretrain_loss_decreases_on_wikitext2_train():
    torch.manual_seed(42)
    result = cogsyndelta.poc.train.train_latent_vae_on_public_split(
        input_dim=768, steps=20, batch_size=8
    )
    assert result['last_loss'] < result['first_loss']
    assert result['split'] == 'train'