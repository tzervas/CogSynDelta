import pytest
from cogsyndelta.poc.train import train_latent_vae_on_public_split

@pytest.mark.cpu
def test_latent_vae_pretrain_loss_decreases_on_wikitext2_train():
    config = {
        'seed': 42,
        'steps': 20,
        'batch_size': 8,
        'input_dim': 768  # Adjust this to match your PoC
    }
    result = train_latent_vae_on_public_split(config)
    assert result['last_loss'] < result['first_loss']
    assert result['split'] == 'train'