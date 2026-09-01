import pytest
import torch
from cogsyndelta.poc.train import train_latent_vae_on_public_split

@pytest.mark.cpu
def test_latent_vae_pretrain_loss_decreases_on_wikitext2_train():
    torch.manual_seed(42)
    config = {
        'dataset': 'Salesforce/wikitext',
        'config': 'wikitext-2-raw-v1',
        'split': 'train',
        'revision': 'b08601e04326c79dfdd32d625aee71d232d685c3',
        'input_dim': 1024,  # Example input dimension
        'batch_size': 8,
        'steps': 20
    }
    result = train_latent_vae_on_public_split(config)
    assert result['last_loss'] < result['first_loss']
    assert result['split'] == 'train'