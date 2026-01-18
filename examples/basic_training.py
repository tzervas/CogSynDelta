"""
Basic CogSynDelta Example: Training a Simple Model

This example demonstrates:
1. Loading the integrated system
2. Training on MNIST dataset
3. Monitoring performance metrics
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from cogsyndelta.core.integrated_system import CogSynDeltaSystem
from cogsyndelta.core.pcn_vae_gan import PCN_VAE_GAN
import yaml


def main():
    """Main training loop."""
    print("="*60)
    print("CogSynDelta Basic Training Example")
    print("="*60)
    
    # Configuration
    config = {
        'embed_dim': 512,
        'memory': {
            'active_capacity': 1000,
            'short_term_capacity': 5000,
            'long_term_capacity': 50000
        },
        'model': {
            'input_dim': 784,  # 28x28 MNIST
            'hidden_dim': 256,
            'latent_dim': 64
        }
    }
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )
    
    # Initialize model
    print("\nInitializing PCN-VAE-GAN model...")
    model = PCN_VAE_GAN(config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    print("\nStarting training...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon, mu, logvar, pred_error = model(data)
            
            # Compute losses
            recon_loss = F.mse_loss(recon, data)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            pcn_loss = pred_error.mean()
            
            loss = recon_loss + 0.001 * kl_loss + 0.1 * pcn_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == '__main__':
    main()
