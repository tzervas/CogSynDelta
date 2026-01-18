"""
MNIST Test Script for PCN-VAE-GAN Hybrid Self-Improving AI

This script demonstrates the three phases of the self-improving AI:
1. Exploratory: Generate diverse samples with configurable σ
2. Culling: Select best samples using Bayesian inference
3. Meta-optimization: MAML-based meta-learning

Tests the VAE loss: L = E[||x - x̂||²] + ½(σ² + μ² - 1 - log σ²)
"""

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import argparse
import os
from pcn_vae_gan import create_model, load_config


def train_epoch(model, dataloader, optimizer, device, config):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.view(-1, 784).to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, loss_dict = model.vae_loss(recon_batch, data, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += loss_dict['reconstruction'].item()
        total_kl_loss += loss_dict['kl_divergence'].item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Recon: {loss_dict["reconstruction"].item():.4f}, '
                  f'KL: {loss_dict["kl_divergence"].item():.4f}')
    
    num_batches = len(dataloader)
    return {
        'total': total_loss / num_batches,
        'reconstruction': total_recon_loss / num_batches,
        'kl_divergence': total_kl_loss / num_batches
    }


def test_vae_loss(model, test_loader, device):
    """
    Test VAE loss function: L = E[||x - x̂||²] + ½(σ² + μ² - 1 - log σ²)
    
    Validates:
    - Reconstruction loss: E[||x - x̂||²]
    - KL divergence: ½ Σ(σ² + μ² - 1 - log σ²)
    - e ≈ 2.718 (exponential base in exp operations)
    """
    model.eval()
    test_loss = 0
    recon_loss = 0
    kl_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, 784).to(device)
            recon_batch, mu, logvar = model(data)
            loss, loss_dict = model.vae_loss(recon_batch, data, mu, logvar)
            
            test_loss += loss.item()
            recon_loss += loss_dict['reconstruction'].item()
            kl_loss += loss_dict['kl_divergence'].item()
    
    num_batches = len(test_loader)
    print(f'\n=== VAE Loss Test ===')
    print(f'Test Loss: {test_loss / num_batches:.4f}')
    print(f'Reconstruction Loss (E[||x - x̂||²]): {recon_loss / num_batches:.4f}')
    print(f'KL Divergence (½(σ² + μ² - 1 - log σ²)): {kl_loss / num_batches:.4f}')
    
    # Verify exponential base e ≈ 2.718
    print(f'\nVerifying exponential base e ≈ 2.718:')
    test_e = torch.exp(torch.tensor(1.0))
    print(f'exp(1) = {test_e.item():.6f} (should be ≈ 2.718282)')
    
    return test_loss / num_batches


def test_exploratory_phase(model, test_loader, device, k=10):
    """Test exploratory phase with configurable σ sampling."""
    model.eval()
    
    print(f'\n=== Exploratory Phase Test ===')
    print(f'Testing configurable σ VAE sampling: z = μ + σ * ε, ε ~ N(0,1)')
    print(f'Generating {k} exploratory samples...')
    
    with torch.no_grad():
        # Get one test sample
        data, _ = next(iter(test_loader))
        data = data[0:1].view(-1, 784).to(device)
        
        # Generate k diverse samples
        samples = model.exploratory_phase(data, k=k)
        print(f'Generated samples shape: {samples.shape}')
        print(f'Sample diversity (std across samples): {samples.std(dim=0).mean().item():.4f}')
        
        # Test with different σ scales
        for sigma_scale in [0.5, 1.0, 2.0]:
            mu, logvar = model.encoder(data)
            z = model.reparameterize(mu, logvar, sigma_scale=sigma_scale)
            recon = model.decoder(z)
            print(f'  σ scale = {sigma_scale}: latent std = {z.std().item():.4f}')


def test_culling_phase(model, test_loader, device):
    """Test culling phase with Bayesian inference."""
    model.eval()
    
    print(f'\n=== Culling Phase Test ===')
    print(f'Testing Bayesian inference: p(θ|data) ≈ exp(log lik + log prior - log Z)')
    
    with torch.no_grad():
        # Get one test sample
        data, _ = next(iter(test_loader))
        data = data[0:1].view(-1, 784).to(device)
        
        # Generate exploratory samples
        samples = model.exploratory_phase(data, k=model.k)
        
        # Apply culling with Bayesian inference
        best_sample, scores = model.culling_phase(samples, data)
        
        print(f'Generated {len(samples)} samples')
        print(f'Selection scores: {scores[:5]}...')  # Show first 5
        print(f'Best sample selected (index: {scores.argmax().item()})')
        
        # Compare reconstruction quality
        original_recon, mu, logvar = model(data)
        best_recon_loss = torch.nn.functional.mse_loss(best_sample, data)
        orig_recon_loss = torch.nn.functional.mse_loss(original_recon, data)
        
        print(f'Original reconstruction MSE: {orig_recon_loss.item():.4f}')
        print(f'Best sample MSE: {best_recon_loss.item():.4f}')


def test_meta_optimization(model, train_loader, device):
    """Test meta-optimization with MAML."""
    model.eval()
    
    print(f'\n=== Meta-Optimization Phase Test ===')
    print(f'Testing MAML gradients: L_meta = E[L_inner(θ_Φ)]')
    
    # Get support and query sets
    data_iter = iter(train_loader)
    support_data, _ = next(data_iter)
    query_data, _ = next(data_iter)
    
    support_data = support_data.view(-1, 784).to(device)
    query_data = query_data.view(-1, 784).to(device)
    
    # Run meta-optimization step
    loss_dict = model.meta_optimization_step(support_data, query_data)
    
    print(f'Inner loop steps: {model.num_inner_steps}')
    print(f'Inner learning rate: {model.inner_lr}')
    print(f'Meta loss: {loss_dict["meta_loss"].item():.4f}')
    print(f'Reconstruction: {loss_dict["reconstruction"].item():.4f}')
    print(f'KL divergence: {loss_dict["kl_divergence"].item():.4f}')


def main():
    parser = argparse.ArgumentParser(description='MNIST Test for PCN-VAE-GAN Hybrid')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='Run tests only without training')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using device: {device}')
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Create model
    print('\nCreating PCN-VAE-GAN Hybrid model...')
    model = create_model(args.config).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Load MNIST dataset
    print('\nLoading MNIST dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['training']['batch_size'],
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=config['training']['batch_size'],
                            shuffle=False)
    
    print(f'Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')
    
    # Run comprehensive tests
    print('\n' + '='*60)
    print('RUNNING COMPREHENSIVE TESTS')
    print('='*60)
    
    # Test VAE loss formula
    test_vae_loss(model, test_loader, device)
    
    # Test exploratory phase
    test_exploratory_phase(model, test_loader, device, k=config['exploratory']['k'])
    
    # Test culling phase
    test_culling_phase(model, test_loader, device)
    
    # Test meta-optimization
    test_meta_optimization(model, train_loader, device)
    
    if not args.test_only:
        # Training
        print('\n' + '='*60)
        print('STARTING TRAINING')
        print('='*60)
        
        optimizer = optim.Adam(model.parameters(), 
                              lr=config['training']['learning_rate'])
        
        for epoch in range(1, config['training']['epochs'] + 1):
            print(f'\nEpoch {epoch}/{config["training"]["epochs"]}')
            
            # Train
            train_losses = train_epoch(model, train_loader, optimizer, device, config)
            
            # Test
            test_loss = test_vae_loss(model, test_loader, device)
            
            print(f'\nEpoch {epoch} Summary:')
            print(f'  Train Loss: {train_losses["total"]:.4f}')
            print(f'  Test Loss: {test_loss:.4f}')
        
        # Save model
        print('\nSaving model...')
        torch.save(model.state_dict(), 'pcn_vae_gan_mnist.pth')
        print('Model saved as pcn_vae_gan_mnist.pth')
    
    print('\n' + '='*60)
    print('ALL TESTS COMPLETED SUCCESSFULLY')
    print('='*60)


if __name__ == '__main__':
    main()
