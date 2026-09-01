# test_poc_region_pretrain.py

import pytest
from cogsyndelta.poc.train import train_latent_vae_on_public_split

@pytest.mark.cpu
def test_latent_vae_pretrain_loss_decreases_on_wikitext2_train():
    # Set the random seed for reproducibility
    import torch
    torch.manual_seed(42)
    
    # Define the configuration for the test
    config = {
        'input_dim': 1024,  # Example input dimension, adjust as needed
        'steps': 20,
        'batch_size': 8,
        'split': 'train'
    }
    
    # Call the helper function to train the LatentVAE model
    result = train_latent_vae_on_public_split(config)
    
    # Assert that the last loss is less than the first loss
    assert result['last_loss'] < result['first_loss'], "The loss did not decrease after training."
    
    # Assert that the split used is 'train'
    assert result['split'] == 'train', "The split used is not 'train'."
