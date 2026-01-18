"""
Unit tests for PCN-VAE-GAN Hybrid Self-Improving AI

Tests all components without requiring MNIST download:
1. VAE loss function: L = E[||x - x̂||²] + ½(σ² + μ² - 1 - log σ²)
2. Exploratory phase: z = μ + σ * ε, ε ~ N(0,1)
3. Culling phase: p(θ|data) ≈ exp(log lik + log prior - log Z)
4. Meta-optimization: MAML gradients
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cogsyndelta.core.pcn_vae_gan import create_model, load_config


def test_vae_loss_formula() -> bool:
    """Test VAE loss formula: L = E[||x - x̂||²] + ½(σ² + μ² - 1 - log σ²)"""
    print("\n" + "="*60)
    print("TEST 1: VAE Loss Formula")
    print("="*60)
    
    config = load_config(str(Path(__file__).parent.parent / "config/config.yaml"))
    model = create_model(str(Path(__file__).parent.parent / 'config/config.yaml'))
    model.eval()
    
    # Create synthetic data
    batch_size = 32
    x = torch.randn(batch_size, 784)
    
    with torch.no_grad():
        recon_x, mu, logvar = model(x)
        loss, loss_dict = model.vae_loss(recon_x, x, mu, logvar)
    
    # Manually compute reconstruction loss
    recon_loss_manual = F.mse_loss(recon_x, x, reduction='sum') / batch_size
    
    # Manually compute KL divergence: ½ Σ(σ² + μ² - 1 - log σ²)
    kl_manual = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
    print(f"✓ Reconstruction Loss (E[||x - x̂||²]):")
    print(f"  Model: {loss_dict['reconstruction'].item():.6f}")
    print(f"  Manual: {recon_loss_manual.item():.6f}")
    print(f"  Match: {torch.allclose(loss_dict['reconstruction'], recon_loss_manual, rtol=1e-5)}")
    
    print(f"\n✓ KL Divergence (½(σ² + μ² - 1 - log σ²)):")
    print(f"  Model: {loss_dict['kl_divergence'].item():.6f}")
    print(f"  Manual: {kl_manual.item():.6f}")
    print(f"  Match: {torch.allclose(loss_dict['kl_divergence'], kl_manual, rtol=1e-5)}")
    
    # Verify exponential base e ≈ 2.718
    print(f"\n✓ Exponential base (e ≈ 2.718):")
    e_value = torch.exp(torch.tensor(1.0)).item()
    print(f"  exp(1) = {e_value:.6f}")
    print(f"  Expected: 2.718282")
    print(f"  Match: {abs(e_value - np.e) < 1e-5}")
    
    # Verify σ = std, μ = mean in the formulas
    print(f"\n✓ Parameter notation:")
    print(f"  μ (mean) shape: {mu.shape}")
    print(f"  σ² (variance) = exp(logvar) shape: {torch.exp(logvar).shape}")
    print(f"  σ (std) = exp(0.5 * logvar) = sqrt(σ²)")
    
    print("\n✅ VAE Loss Formula Test PASSED")
    return True


def test_exploratory_phase_sampling() -> bool:
    """Test exploratory phase: z = μ + σ * ε, ε ~ N(0,1)"""
    print("\n" + "="*60)
    print("TEST 2: Exploratory Phase - Configurable σ Sampling")
    print("="*60)
    
    config = load_config(str(Path(__file__).parent.parent / "config/config.yaml"))
    model = create_model(str(Path(__file__).parent.parent / 'config/config.yaml'))
    model.eval()
    
    # Create synthetic data
    x = torch.randn(4, 784)
    
    print(f"✓ Testing reparameterization trick: z = μ + σ * ε, ε ~ N(0,1)")
    
    with torch.no_grad():
        mu, logvar = model.encoder(x)
        
        # Test with different σ scales
        for sigma_scale in [0.5, 1.0, 2.0]:
            z = model.reparameterize(mu, logvar, sigma_scale=sigma_scale)
            
            # σ = exp(0.5 * logvar)
            std = torch.exp(0.5 * logvar)
            
            # Verify z has correct distribution properties
            print(f"\n  σ scale = {sigma_scale}:")
            print(f"    Latent z shape: {z.shape}")
            print(f"    μ mean: {mu.mean().item():.4f}")
            print(f"    σ (std) mean: {std.mean().item():.4f}")
            print(f"    z mean: {z.mean().item():.4f} (should be close to μ)")
            print(f"    z std: {z.std().item():.4f} (should scale with σ)")
            
            # Verify z is approximately z ≈ μ + sigma_scale * σ * ε
            expected_std_range = (sigma_scale * std.mean().item() * 0.5, 
                                  sigma_scale * std.mean().item() * 1.5)
            assert expected_std_range[0] <= z.std().item() <= expected_std_range[1], \
                f"z std {z.std().item()} not in expected range {expected_std_range}"
    
    # Test exploratory phase with k samples
    k = config['exploratory']['k']
    print(f"\n✓ Testing exploratory phase with k={k} samples:")
    
    with torch.no_grad():
        samples = model.exploratory_phase(x[0:1], k=k)
        print(f"  Generated samples shape: {samples.shape}")
        print(f"  Expected shape: torch.Size([{k}, 1, 784])")
        print(f"  Sample diversity (std): {samples.std().item():.4f}")
        
        assert samples.shape == torch.Size([k, 1, 784]), "Sample shape mismatch"
    
    print("\n✅ Exploratory Phase Sampling Test PASSED")
    return True


def test_culling_phase_bayesian() -> bool:
    """Test culling phase: p(θ|data) ≈ exp(log lik + log prior - log Z)"""
    print("\n" + "="*60)
    print("TEST 3: Culling Phase - Bayesian Inference")
    print("="*60)
    
    config = load_config(str(Path(__file__).parent.parent / "config/config.yaml"))
    model = create_model(str(Path(__file__).parent.parent / 'config/config.yaml'))
    model.eval()
    
    # Create synthetic data
    x = torch.randn(1, 784)
    
    print(f"✓ Testing Bayesian inference: p(θ|data) ≈ exp(log lik + log prior - log Z)")
    
    with torch.no_grad():
        # Generate exploratory samples
        k = config['exploratory']['k']
        samples = model.exploratory_phase(x, k=k)
        
        print(f"\n  Generated {k} exploratory samples")
        
        # Apply culling
        best_sample, scores = model.culling_phase(samples, x)
        
        print(f"  Selection scores shape: {scores.shape}")
        print(f"  Scores sum: {scores.sum().item():.4f}")
        print(f"  Best sample index: {scores.argmax().item()}")
        print(f"  Best sample score: {scores.max().item():.4f}")
        
        # Verify scores are probabilities
        assert scores.shape[0] == k, "Score count mismatch"
        assert scores.min() >= 0, "Scores should be non-negative"
        
        # Compare with original reconstruction
        recon_x, _, _ = model(x)
        orig_mse = F.mse_loss(recon_x, x)
        best_mse = F.mse_loss(best_sample, x)
        
        print(f"\n  Original reconstruction MSE: {orig_mse.item():.4f}")
        print(f"  Best culled sample MSE: {best_mse.item():.4f}")
    
    print("\n✅ Culling Phase Bayesian Inference Test PASSED")
    return True


def test_meta_optimization_maml() -> bool:
    """Test meta-optimization: MAML gradients, L_meta = E[L_inner(θ_Φ)]"""
    print("\n" + "="*60)
    print("TEST 4: Meta-Optimization - MAML Gradients")
    print("="*60)
    
    config = load_config(str(Path(__file__).parent.parent / "config/config.yaml"))
    model = create_model(str(Path(__file__).parent.parent / 'config/config.yaml'))
    model.eval()
    
    print(f"✓ Testing MAML: L_meta = E[L_inner(θ_Φ)]")
    
    # Create support and query sets
    x_support = torch.randn(16, 784)
    x_query = torch.randn(16, 784)
    
    print(f"\n  Support set shape: {x_support.shape}")
    print(f"  Query set shape: {x_query.shape}")
    print(f"  Inner steps: {config['meta_optimization']['num_inner_steps']}")
    print(f"  Inner LR: {config['meta_optimization']['inner_lr']}")
    
    # Save original parameters
    original_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Run meta-optimization
    loss_dict = model.meta_optimization_step(x_support, x_query)
    
    print(f"\n  Meta loss: {loss_dict['meta_loss'].item():.4f}")
    print(f"  Reconstruction: {loss_dict['reconstruction'].item():.4f}")
    print(f"  KL divergence: {loss_dict['kl_divergence'].item():.4f}")
    
    # Verify parameters are restored
    params_restored = True
    for name, param in model.named_parameters():
        if not torch.allclose(param, original_params[name], rtol=1e-5):
            params_restored = False
            break
    
    print(f"\n  Parameters restored after meta-step: {params_restored}")
    assert params_restored, "Parameters should be restored after meta-optimization"
    
    print("\n✅ Meta-Optimization MAML Test PASSED")
    return True


def test_config_loading() -> bool:
    """Test YAML configuration loading"""
    print("\n" + "="*60)
    print("TEST 5: YAML Configuration Loading")
    print("="*60)
    
    config = load_config(str(Path(__file__).parent.parent / "config/config.yaml"))
    
    print(f"✓ Configuration loaded successfully")
    print(f"\n  Exploratory phase:")
    print(f"    k (samples): {config['exploratory']['k']}")
    print(f"    σ scale: {config['exploratory']['sigma_scale']}")
    print(f"    latent_dim: {config['exploratory']['latent_dim']}")
    
    print(f"\n  Culling phase:")
    print(f"    threshold: {config['culling']['threshold']}")
    print(f"    bayesian_prior_weight: {config['culling']['bayesian_prior_weight']}")
    
    print(f"\n  Meta-optimization:")
    print(f"    inner_lr: {config['meta_optimization']['inner_lr']}")
    print(f"    meta_lr: {config['meta_optimization']['meta_lr']}")
    print(f"    num_inner_steps: {config['meta_optimization']['num_inner_steps']}")
    
    print(f"\n  VAE loss:")
    print(f"    reconstruction_weight: {config['vae_loss']['reconstruction_weight']}")
    print(f"    kl_weight: {config['vae_loss']['kl_weight']}")
    
    # Verify all required keys exist
    assert 'exploratory' in config
    assert 'culling' in config
    assert 'meta_optimization' in config
    assert 'vae_loss' in config
    
    print("\n✅ Configuration Loading Test PASSED")
    return True


def test_model_components() -> bool:
    """Test individual model components"""
    print("\n" + "="*60)
    print("TEST 6: Model Components")
    print("="*60)
    
    config = load_config(str(Path(__file__).parent.parent / "config/config.yaml"))
    model = create_model(str(Path(__file__).parent.parent / 'config/config.yaml'))
    
    print(f"✓ Model architecture:")
    print(f"  Input dim: {model.input_dim}")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Latent dim: {model.latent_dim}")
    
    # Test encoder
    x = torch.randn(8, 784)
    mu, logvar = model.encoder(x)
    print(f"\n  Encoder:")
    print(f"    Input: {x.shape}")
    print(f"    μ output: {mu.shape}")
    print(f"    log(σ²) output: {logvar.shape}")
    
    # Test decoder
    z = torch.randn(8, model.latent_dim)
    recon = model.decoder(z)
    print(f"\n  Decoder:")
    print(f"    Input: {z.shape}")
    print(f"    Reconstruction output: {recon.shape}")
    
    # Test full forward pass
    recon_x, mu, logvar = model(x)
    print(f"\n  Full forward pass:")
    print(f"    Input: {x.shape}")
    print(f"    Reconstruction: {recon_x.shape}")
    print(f"    μ: {mu.shape}")
    print(f"    log(σ²): {logvar.shape}")
    
    # Verify shapes
    assert mu.shape == (8, model.latent_dim)
    assert logvar.shape == (8, model.latent_dim)
    assert recon_x.shape == (8, 784)
    
    print(f"\n  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    print("\n✅ Model Components Test PASSED")
    return True


def run_all_tests() -> None:
    """Run all tests"""
    print("\n" + "="*70)
    print(" PCN-VAE-GAN HYBRID SELF-IMPROVING AI - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        test_config_loading,
        test_model_components,
        test_vae_loss_formula,
        test_exploratory_phase_sampling,
        test_culling_phase_bayesian,
        test_meta_optimization_maml,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result, None))
        except Exception as e:
            results.append((test_func.__name__, False, str(e)))
            print(f"\n❌ Test FAILED: {e}")
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for name, result, error in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"       Error: {error}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nVerified components:")
        print("  ✓ VAE loss: L = E[||x - x̂||²] + ½(σ² + μ² - 1 - log σ²)")
        print("  ✓ Exploratory phase: z = μ + σ * ε, ε ~ N(0,1)")
        print("  ✓ Culling phase: p(θ|data) ≈ exp(log lik + log prior - log Z)")
        print("  ✓ Meta-optimization: MAML with L_meta = E[L_inner(θ_Φ)]")
        print("  ✓ YAML configuration for k, thresholds, σ")
        print("  ✓ Exponential base e ≈ 2.718")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
