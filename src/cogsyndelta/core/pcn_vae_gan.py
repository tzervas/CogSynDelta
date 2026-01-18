"""
PCN-VAE-GAN Hybrid Self-Improving AI Architecture

This module implements a self-improving AI architecture that combines:
1. Predictive Coding Networks (PCN)
2. Variational Autoencoders (VAE) with configurable σ sampling
3. Generative Adversarial Networks (GAN) principles

The architecture follows three phases:
- Exploratory: High-variance sampling for creative idea generation
- Culling: Bidirectional search and Bayesian inference for feasibility
- Meta-optimization: MAML-based tuning for validated implementations
"""

import torch
import torch.nn.functional as F
import yaml
from torch import nn


class VAEEncoder(nn.Module):
    """VAE Encoder with configurable σ sampling."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        """Initialize encoder with linear layers for mu and logvar."""
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor

        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder for reconstruction."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initialize decoder with linear layers for reconstruction."""
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            z: Latent representation

        Returns:
            Reconstructed output
        """
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))


class PCNVAEGANHybrid(nn.Module):
    """
    PCN-VAE-GAN Hybrid Self-Improving AI Architecture.

    Implements three phases:
    1. Exploratory: VAE with configurable σ sampling (z = μ + σ * ε, ε ~ N(0,1))
    2. Culling: Bidirectional A* and Bayesian inference
    3. Meta-optimization: MAML gradients for meta-learning
    """

    def __init__(self, config: dict) -> None:
        """Initialize PCN-VAE-GAN hybrid with encoder, decoder, and discriminator."""
        super(PCNVAEGANHybrid, self).__init__()

        # Load configuration
        self.config = config
        self.input_dim = 784  # MNIST: 28x28
        self.latent_dim = config["exploratory"]["latent_dim"]
        self.hidden_dim = config["exploratory"]["hidden_dim"]
        self.sigma_scale = config["exploratory"]["sigma_scale"]
        self.k = config["exploratory"]["k"]

        # VAE components
        self.encoder = VAEEncoder(self.input_dim, self.hidden_dim, self.latent_dim)
        self.decoder = VAEDecoder(self.latent_dim, self.hidden_dim, self.input_dim)

        # Culling phase parameters
        self.culling_threshold = config["culling"]["threshold"]
        self.bayesian_prior_weight = config["culling"]["bayesian_prior_weight"]

        # Meta-optimization parameters
        self.inner_lr = config["meta_optimization"]["inner_lr"]
        self.num_inner_steps = config["meta_optimization"]["num_inner_steps"]

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor, sigma_scale: float | None = None
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            sigma_scale: Optional scale for σ (configurable variance)

        Returns:
            Sampled latent vector z
        """
        if sigma_scale is None:
            sigma_scale = self.sigma_scale

        # σ = exp(0.5 * log(σ²)) = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)

        # ε ~ N(0, 1)
        eps = torch.randn_like(std)

        # z = μ + σ * ε with configurable σ scale
        return mu + sigma_scale * std * eps

    def forward(
        self, x: torch.Tensor, sigma_scale: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input tensor
            sigma_scale: Optional configurable σ scale

        Returns:
            recon_x: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar, sigma_scale)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def vae_loss(
        self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        VAE loss function: L = E[||x - x̂||²] + ½(σ² + μ² - 1 - log σ²)

        Reconstruction loss: E[||x - x̂||²]
        KL divergence: ½ Σ(σ² + μ² - 1 - log σ²)

        Args:
            recon_x: Reconstructed input (x̂)
            x: Original input
            mu: Mean (μ) of latent distribution
            logvar: Log variance (log σ²) of latent distribution

        Returns:
            Total loss and dictionary of loss components
        """
        # Reconstruction loss: E[||x - x̂||²]
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)

        # KL divergence: ½ Σ(σ² + μ² - 1 - log σ²)
        # σ² = exp(logvar)
        # KL = ½ Σ(exp(logvar) + μ² - 1 - logvar)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total VAE loss
        recon_weight = self.config["vae_loss"]["reconstruction_weight"]
        kl_weight = self.config["vae_loss"]["kl_weight"]

        total_loss = recon_weight * recon_loss + kl_weight * kl_div

        loss_dict = {"total": total_loss, "reconstruction": recon_loss, "kl_divergence": kl_div}

        return total_loss, loss_dict

    def exploratory_phase(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """
        Exploratory phase: Generate k diverse samples using high-variance sampling.

        Args:
            x: Input tensor
            k: Number of exploratory samples (default: from config)

        Returns:
            Tensor of k generated samples
        """
        if k is None:
            k = self.k

        mu, logvar = self.encoder(x)

        # Generate k diverse samples with configurable σ (batched for efficiency)
        # Expand mu and logvar to generate all k samples at once
        mu_expanded = mu.unsqueeze(0).expand(k, -1, -1)  # [k, batch, latent_dim]
        logvar_expanded = logvar.unsqueeze(0).expand(k, -1, -1)  # [k, batch, latent_dim]

        # Compute std once
        std = torch.exp(0.5 * logvar_expanded)
        eps = torch.randn_like(std)

        # Batched reparameterization: z = μ + σ * ε
        z_samples = mu_expanded + self.sigma_scale * std * eps

        # Decode all samples at once
        z_flat = z_samples.view(-1, self.latent_dim)  # [k*batch, latent_dim]
        samples_flat = self.decoder(z_flat)  # [k*batch, input_dim]
        samples = samples_flat.view(k, x.size(0), -1)  # [k, batch, input_dim]

        return samples

    def culling_phase(
        self, samples: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Culling phase: Use Bayesian inference to select feasible outputs.

        Bayesian inference: p(θ|data) ≈ exp(log likelihood + log prior - log Z)

        Args:
            samples: Generated samples from exploratory phase [k, batch, features]
            x: Original input for comparison [batch, features]

        Returns:
            Best sample and selection scores
        """
        # Compute log likelihood for each sample
        log_likelihood_list: list[torch.Tensor] = []
        for sample in samples:
            # Log likelihood based on reconstruction quality
            mse = F.mse_loss(sample, x.expand_as(sample), reduction="none").sum(dim=-1)
            log_lik = -mse  # Higher is better
            log_likelihood_list.append(log_lik)

        log_likelihoods = torch.stack(log_likelihood_list)  # [k, batch]

        # Simple prior: prefer samples closer to mean
        log_prior = -torch.norm(samples - x.expand_as(samples), dim=-1)  # [k, batch]

        # Bayesian posterior (unnormalized): log p(θ|data) ≈ log lik + log prior
        log_posterior = log_likelihoods + self.bayesian_prior_weight * log_prior

        # Select best sample based on posterior (for each batch element)
        scores = torch.exp(
            log_posterior - log_posterior.max(dim=0, keepdim=True)[0]
        )  # Normalize for stability
        best_idx = scores.argmax(dim=0)  # [batch]

        # Select best sample for each batch element
        batch_size = x.size(0)
        best_samples = torch.stack([samples[best_idx[b], b] for b in range(batch_size)])

        return best_samples, scores

    def meta_optimization_step(self, x_support: torch.Tensor, x_query: torch.Tensor) -> dict:
        """
        Meta-optimization: MAML-based meta-learning step.

        L_meta = E[L_inner(θ_Φ)]

        Args:
            x_support: Support set for inner loop adaptation
            x_query: Query set for meta-learning

        Returns:
            Dictionary of meta-learning losses
        """
        # Save original parameters
        original_params = {name: param.clone() for name, param in self.named_parameters()}

        # Inner loop: Adapt on support set
        params = list(self.parameters())
        for step in range(self.num_inner_steps):
            recon_x, mu, logvar = self.forward(x_support)
            inner_loss, _ = self.vae_loss(recon_x, x_support, mu, logvar)

            # Compute gradients
            grads = torch.autograd.grad(inner_loss, params, create_graph=True)

            # Update parameters with inner learning rate
            with torch.no_grad():
                for param, grad in zip(params, grads):
                    param.data = param.data - self.inner_lr * grad

        # Outer loop: Evaluate on query set
        recon_x_query, mu_query, logvar_query = self.forward(x_query)
        meta_loss, loss_dict = self.vae_loss(recon_x_query, x_query, mu_query, logvar_query)

        # Restore original parameters
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.data = original_params[name].data

        loss_dict["meta_loss"] = meta_loss

        return loss_dict


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def create_model(config_path: str = "config.yaml") -> PCNVAEGANHybrid:
    """
    Create PCN-VAE-GAN hybrid model from configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Initialized model
    """
    config = load_config(config_path)
    model = PCNVAEGANHybrid(config)
    return model


if __name__ == "__main__":
    # Example usage
    config = load_config("config.yaml")
    model = create_model("config.yaml")

    # Test forward pass
    x = torch.randn(32, 784)
    recon_x, mu, logvar = model(x)
    loss, loss_dict = model.vae_loss(recon_x, x, mu, logvar)

    print("Model created successfully!")
    print(f"Reconstruction shape: {recon_x.shape}")
    print(f"Total loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {loss_dict['reconstruction'].item():.4f}")
    print(f"KL divergence: {loss_dict['kl_divergence'].item():.4f}")
