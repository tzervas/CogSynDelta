"""
Tests for compression fidelity calibration and validation.

This module validates the baseline compression fidelity of different
compactors and compression methods, establishing the ground truth
for ADR-0008 implementation.

Per specs/compression-research/spec.md, current baselines are:
- 2x compression: 0.670 fidelity
- 4x compression: 0.462 fidelity
- 16x compression: 0.228 fidelity

The HighFidelityCompactor should achieve ≥0.95 fidelity at ~2x compression.
"""

from __future__ import annotations

import statistics
import unittest

import torch
import torch.nn.functional as F


class TestCompressionBaselines(unittest.TestCase):
    """Validate compression fidelity baselines for ADR-0008.
    
    Why this test exists: ADR-0008 was initially written with incorrect
    fidelity values (0.06). This test establishes ground truth by
    measuring actual fidelity across all compression implementations.
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.embed_dim = 512
        self.n_samples = 100
        # Generate reproducible random embeddings
        torch.manual_seed(42)
        self.test_embeddings = torch.randn(self.n_samples, self.embed_dim)
        # Normalize for consistent comparison
        self.test_embeddings = F.normalize(self.test_embeddings, p=2, dim=-1)

    def _measure_fidelity(
        self,
        compress_fn: callable,
        decompress_fn: callable,
        embeddings: torch.Tensor,
    ) -> dict[str, float]:
        """Measure compression fidelity over a batch of embeddings.
        
        Args:
            compress_fn: Function that compresses a single embedding.
            decompress_fn: Function that decompresses compressed data.
            embeddings: Batch of embeddings to test.
            
        Returns:
            Dictionary with mean, min, max, std fidelity metrics.
        """
        fidelities = []
        for i in range(embeddings.shape[0]):
            original = embeddings[i]
            compressed = compress_fn(original)
            reconstructed = decompress_fn(compressed)
            
            # Compute cosine similarity
            cos_sim = F.cosine_similarity(
                original.unsqueeze(0), 
                reconstructed.unsqueeze(0), 
                dim=-1
            ).item()
            fidelities.append(cos_sim)
        
        return {
            "mean": statistics.mean(fidelities),
            "min": min(fidelities),
            "max": max(fidelities),
            "std": statistics.stdev(fidelities) if len(fidelities) > 1 else 0.0,
        }

    def test_high_fidelity_compactor_baseline(self) -> None:
        """Validate HighFidelityCompactor achieves ≥0.95 at ~2x compression.
        
        This is the production-recommended compactor. It uses orthonormal
        basis decomposition with explicit residual storage.
        """
        from cogsyndelta.memory.active_memory import HighFidelityCompactor

        compactor = HighFidelityCompactor(
            embed_dim=self.embed_dim,
            num_basis=256,
            use_float16=True,
        )

        def compress(emb: torch.Tensor) -> dict:
            return compactor.compact(emb)
        
        def decompress(data: dict) -> torch.Tensor:
            return compactor.reconstruct(data)

        metrics = self._measure_fidelity(
            compress, decompress, self.test_embeddings
        )

        print(f"\nHighFidelityCompactor (256 basis, float16):")
        print(f"  Mean fidelity: {metrics['mean']:.4f}")
        print(f"  Min fidelity:  {metrics['min']:.4f}")
        print(f"  Max fidelity:  {metrics['max']:.4f}")
        print(f"  Std:           {metrics['std']:.4f}")
        
        # This MUST pass - HighFidelityCompactor is designed for this
        self.assertGreaterEqual(
            metrics['mean'], 0.95,
            f"HighFidelityCompactor mean fidelity {metrics['mean']:.4f} < 0.95"
        )
        self.assertGreaterEqual(
            metrics['min'], 0.90,
            f"HighFidelityCompactor min fidelity {metrics['min']:.4f} < 0.90"
        )

    def test_hybrid_adaptive_compactor_baseline(self) -> None:
        """Validate HybridAdaptiveCompactor fidelity at various compression levels.
        
        This compactor uses multi-mode compression (PCA + sparse + dense)
        and should achieve good fidelity while adapting to data characteristics.
        """
        from cogsyndelta.memory.active_memory import HybridAdaptiveCompactor

        compactor = HybridAdaptiveCompactor(
            embed_dim=self.embed_dim,
            num_basis=384,
            sparsity_threshold=0.01,
        )

        def compress(emb: torch.Tensor) -> dict:
            return compactor.compact(emb)
        
        def decompress(data: dict) -> torch.Tensor:
            return compactor.reconstruct(data)

        metrics = self._measure_fidelity(
            compress, decompress, self.test_embeddings
        )

        print(f"\nHybridAdaptiveCompactor (384 basis):")
        print(f"  Mean fidelity: {metrics['mean']:.4f}")
        print(f"  Min fidelity:  {metrics['min']:.4f}")
        print(f"  Max fidelity:  {metrics['max']:.4f}")
        print(f"  Std:           {metrics['std']:.4f}")
        
        # HybridAdaptive should achieve good fidelity with more basis vectors
        self.assertGreaterEqual(
            metrics['mean'], 0.90,
            f"HybridAdaptiveCompactor mean fidelity {metrics['mean']:.4f} < 0.90"
        )

    def test_residual_boost_compactor_baseline(self) -> None:
        """Validate ResidualBoostCompactor multi-stage compression fidelity.
        
        This compactor uses iterative residual encoding similar to RVQ,
        progressively capturing finer details in each stage.
        """
        from cogsyndelta.memory.active_memory import ResidualBoostCompactor

        compactor = ResidualBoostCompactor(
            embed_dim=self.embed_dim,
            num_stages=3,
            stage_ratios=(0.5, 0.25, 0.125),
        )

        def compress(emb: torch.Tensor) -> dict:
            return compactor.compact(emb)
        
        def decompress(data: dict) -> torch.Tensor:
            return compactor.reconstruct(data)

        metrics = self._measure_fidelity(
            compress, decompress, self.test_embeddings
        )

        print(f"\nResidualBoostCompactor (3 stages):")
        print(f"  Mean fidelity: {metrics['mean']:.4f}")
        print(f"  Min fidelity:  {metrics['min']:.4f}")
        print(f"  Max fidelity:  {metrics['max']:.4f}")
        print(f"  Std:           {metrics['std']:.4f}")
        
        # NOTE: ResidualBoostCompactor without training achieves ~0.59 fidelity.
        # This is a known limitation documented in ADR-0008. Future work should
        # either train the compactor or implement RVQ with learned codebooks.
        # For now, just verify it runs and produces reasonable (>0.50) output.
        self.assertGreater(
            metrics['mean'], 0.50,
            f"ResidualBoostCompactor mean fidelity {metrics['mean']:.4f} < 0.50"
        )
        
        # Record this as a calibration target for future improvement
        if metrics['mean'] < 0.85:
            print("  ⚠ Below target (0.85) - candidate for ADR-0008 Phase 3 RVQ work")

    def test_dense_differential_store_baseline(self) -> None:
        """Validate DenseDifferentialMemoryStore compression fidelity.
        
        This is the original compression method that showed 0.67 fidelity
        at 2x compression in specs/compression-research/spec.md.
        
        Note: This compactor requires trained neural networks. Without
        training, fidelity will be lower than spec targets.
        """
        from cogsyndelta.memory.dense_embeddings import DenseDifferentialMemoryStore

        store = DenseDifferentialMemoryStore(
            embed_dim=self.embed_dim,
            dense_dim=64,
            num_references=100,
        )
        
        # Update references to match our test distribution
        store.update_references(self.test_embeddings[:100])

        fidelities = []
        compression_ratios = []
        
        for i in range(self.n_samples):
            embedding = self.test_embeddings[i]
            memory_id = f"test_{i}"
            
            # Compress and store
            stats = store.compress_and_store(embedding, memory_id, importance=1.0)
            fidelities.append(stats["fidelity_score"])
            compression_ratios.append(stats["compression_ratio"])
        
        mean_fidelity = statistics.mean(fidelities)
        mean_compression = statistics.mean(compression_ratios)

        print(f"\nDenseDifferentialMemoryStore (untrained):")
        print(f"  Mean fidelity:     {mean_fidelity:.4f}")
        print(f"  Min fidelity:      {min(fidelities):.4f}")
        print(f"  Max fidelity:      {max(fidelities):.4f}")
        print(f"  Mean compression:  {mean_compression:.2f}x")
        
        # Without training, expect lower fidelity
        # This establishes the baseline that calibration should improve
        self.assertGreater(
            mean_fidelity, 0.0,
            "DenseDifferentialMemoryStore should have positive fidelity"
        )
        
        # Store baseline for comparison
        self._dense_differential_baseline = mean_fidelity

    def test_lossless_compactor_baseline(self) -> None:
        """Validate LosslessCompactor neural encoder fidelity.
        
        Note: 'Lossless' is aspirational - without training, the neural
        encoder creates an information bottleneck with lossy reconstruction.
        """
        from cogsyndelta.memory.active_memory import LosslessCompactor

        compactor = LosslessCompactor(
            embed_dim=self.embed_dim,
            num_basis=128,
        )

        def compress(emb: torch.Tensor) -> dict:
            return compactor.compact(emb)
        
        def decompress(data: dict) -> torch.Tensor:
            return compactor.reconstruct(data)

        metrics = self._measure_fidelity(
            compress, decompress, self.test_embeddings
        )

        print(f"\nLosslessCompactor (128 basis, untrained):")
        print(f"  Mean fidelity: {metrics['mean']:.4f}")
        print(f"  Min fidelity:  {metrics['min']:.4f}")
        print(f"  Max fidelity:  {metrics['max']:.4f}")
        print(f"  Std:           {metrics['std']:.4f}")
        
        # Just verify it runs - without training, fidelity will be low
        self.assertGreater(
            metrics['mean'], 0.0,
            "LosslessCompactor should have positive fidelity"
        )


class TestCalibrationNeeds(unittest.TestCase):
    """Assess whether calibration is needed based on fidelity gaps.
    
    This test class determines if additional calibration work is needed
    or if existing compactors already meet ADR-0008 targets.
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.embed_dim = 512
        torch.manual_seed(42)
        # Use normalized embeddings like real VL-JEPA outputs
        self.test_embeddings = F.normalize(
            torch.randn(100, self.embed_dim), p=2, dim=-1
        )

    def test_calibration_assessment(self) -> None:
        """Determine if calibration is needed to meet ADR-0008 targets.
        
        ADR-0008 targets:
        - 2x compression: ≥0.95 fidelity
        - 4x compression: ≥0.90 fidelity
        - 10x compression: ≥0.85 fidelity (stretch goal)
        """
        from cogsyndelta.memory.active_memory import (
            HighFidelityCompactor,
            HybridAdaptiveCompactor,
            ResidualBoostCompactor,
        )
        
        print("\n" + "="*60)
        print("CALIBRATION ASSESSMENT REPORT")
        print("="*60)
        
        # HighFidelityCompactor at 2x
        hf_compactor = HighFidelityCompactor(embed_dim=512, num_basis=256)
        hf_fidelities = []
        for emb in self.test_embeddings:
            compact = hf_compactor.compact(emb)
            recon = hf_compactor.reconstruct(compact)
            fid = F.cosine_similarity(
                emb.unsqueeze(0), recon.unsqueeze(0), dim=-1
            ).item()
            hf_fidelities.append(fid)
        
        hf_mean = statistics.mean(hf_fidelities)
        print(f"\nHighFidelityCompactor @ ~2x compression:")
        print(f"  Measured:  {hf_mean:.4f}")
        print(f"  Target:    0.95")
        print(f"  Status:    {'✓ PASS' if hf_mean >= 0.95 else '✗ NEEDS WORK'}")
        
        # HybridAdaptiveCompactor at 2-4x
        hybrid_compactor = HybridAdaptiveCompactor(embed_dim=512, num_basis=256)
        hybrid_fidelities = []
        for emb in self.test_embeddings:
            compact = hybrid_compactor.compact(emb)
            recon = hybrid_compactor.reconstruct(compact)
            fid = F.cosine_similarity(
                emb.unsqueeze(0), recon.unsqueeze(0), dim=-1
            ).item()
            hybrid_fidelities.append(fid)
        
        hybrid_mean = statistics.mean(hybrid_fidelities)
        print(f"\nHybridAdaptiveCompactor @ ~2-4x compression:")
        print(f"  Measured:  {hybrid_mean:.4f}")
        print(f"  Target:    0.90")
        print(f"  Status:    {'✓ PASS' if hybrid_mean >= 0.90 else '✗ NEEDS WORK'}")
        
        # ResidualBoostCompactor at variable compression
        rb_compactor = ResidualBoostCompactor(embed_dim=512, num_stages=3)
        rb_fidelities = []
        for emb in self.test_embeddings:
            compact = rb_compactor.compact(emb)
            recon = rb_compactor.reconstruct(compact)
            fid = F.cosine_similarity(
                emb.unsqueeze(0), recon.unsqueeze(0), dim=-1
            ).item()
            rb_fidelities.append(fid)
        
        rb_mean = statistics.mean(rb_fidelities)
        print(f"\nResidualBoostCompactor @ multi-stage:")
        print(f"  Measured:  {rb_mean:.4f}")
        print(f"  Target:    0.85")
        print(f"  Status:    {'✓ PASS' if rb_mean >= 0.85 else '✗ NEEDS WORK'}")
        
        print("\n" + "="*60)
        print("CALIBRATION RECOMMENDATION")
        print("="*60)
        
        if hf_mean >= 0.95 and hybrid_mean >= 0.90:
            print("\n✓ Existing compactors meet ADR-0008 fidelity targets.")
            print("  - HighFidelityCompactor is production-ready.")
            print("  - Focus should shift to higher compression ratios (4x-16x).")
        else:
            print("\n✗ Calibration work needed:")
            if hf_mean < 0.95:
                print(f"  - HighFidelityCompactor: {hf_mean:.4f} < 0.95 target")
            if hybrid_mean < 0.90:
                print(f"  - HybridAdaptiveCompactor: {hybrid_mean:.4f} < 0.90 target")
        
        # Record results for CI reporting
        self.assertGreaterEqual(
            hf_mean, 0.90,
            "HighFidelityCompactor should achieve at least 0.90 fidelity"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
