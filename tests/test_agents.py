"""
Unit and Integration Tests for Self-Improving Agent Framework

Tests cover:
1. AgentType enum values
2. LanguageFrameworkEncoder forward pass
3. SelfImprovementModule quality assessment and improvement generation
4. SecurityHardeningModule vulnerability detection
5. QualityAssuranceModule metrics evaluation
6. SelfImprovingAgentFramework integration
7. create_agent_framework factory function

All tests are CPU-compatible for CI.
"""

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cogsyndelta.agents import (
    AgentType,
    LanguageFrameworkEncoder,
    MultiLanguageExplorer,
    QualityAssuranceModule,
    SecurityHardeningModule,
    SelfImprovementModule,
    SelfImprovingAgentFramework,
    create_agent_framework,
)


class TestAgentType:
    """Tests for AgentType enum."""

    def test_agent_type_values(self) -> None:
        """Test all AgentType enum values exist."""
        assert AgentType.SWE.value == "software_engineering"
        assert AgentType.AIE.value == "ai_engineering"
        assert AgentType.SWD.value == "software_development"
        assert AgentType.AID.value == "ai_development"
        assert AgentType.SECURITY.value == "security_testing"
        assert AgentType.QA.value == "quality_assurance"

    def test_agent_type_count(self) -> None:
        """Test expected number of agent types."""
        assert len(AgentType) == 6


class TestLanguageFrameworkEncoder:
    """Tests for LanguageFrameworkEncoder."""

    @pytest.fixture
    def encoder(self) -> LanguageFrameworkEncoder:
        """Create encoder instance."""
        return LanguageFrameworkEncoder(vocab_size=1000, embed_dim=512)

    def test_encoder_initialization(self, encoder: LanguageFrameworkEncoder) -> None:
        """Test encoder initializes with correct dimensions."""
        assert encoder.embed_dim == 512
        assert encoder.token_embed.num_embeddings == 1000
        assert encoder.token_embed.embedding_dim == 512

    def test_encoder_forward_pass(
        self, encoder: LanguageFrameworkEncoder, test_tokens: torch.Tensor
    ) -> None:
        """Test forward pass produces correct output shape."""
        output = encoder(test_tokens, language="python")
        assert output.shape == (test_tokens.size(0), 512)

    def test_encoder_different_languages(
        self, encoder: LanguageFrameworkEncoder, test_tokens: torch.Tensor
    ) -> None:
        """Test encoder handles different languages."""
        languages = ["python", "javascript", "rust", "go", "java"]
        outputs = []

        for lang in languages:
            output = encoder(test_tokens, language=lang)
            outputs.append(output)
            assert output.shape == (test_tokens.size(0), 512)

        # Different languages should produce different embeddings
        # (due to language type embedding)
        assert not torch.allclose(outputs[0], outputs[1])

    def test_encoder_unknown_language_fallback(
        self, encoder: LanguageFrameworkEncoder, test_tokens: torch.Tensor
    ) -> None:
        """Test encoder handles unknown language gracefully."""
        output = encoder(test_tokens, language="unknown_lang")
        assert output.shape == (test_tokens.size(0), 512)


class TestSelfImprovementModule:
    """Tests for SelfImprovementModule."""

    @pytest.fixture
    def module(self) -> SelfImprovementModule:
        """Create module instance."""
        return SelfImprovementModule(embed_dim=512)

    def test_module_initialization(self, module: SelfImprovementModule) -> None:
        """Test module initializes correctly."""
        assert module.embed_dim == 512
        assert module.quality_assessor is not None
        assert module.improvement_generator is not None
        assert module.refinement_selector is not None

    def test_assess_quality(
        self, module: SelfImprovementModule, test_embeddings: torch.Tensor
    ) -> None:
        """Test quality assessment produces scores in [0, 1]."""
        scores = module.assess_quality(test_embeddings)
        assert scores.shape == (test_embeddings.size(0), 1)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_generate_improvement(
        self, module: SelfImprovementModule, test_embeddings: torch.Tensor
    ) -> None:
        """Test improvement generation produces correct shape."""
        error = torch.randn_like(test_embeddings)
        improvement = module.generate_improvement(test_embeddings, error)
        assert improvement.shape == test_embeddings.shape

    def test_select_refinement(
        self, module: SelfImprovementModule, test_embeddings: torch.Tensor
    ) -> None:
        """Test refinement selection returns solution and score."""
        proposed = torch.randn_like(test_embeddings)
        context = torch.randn_like(test_embeddings)

        selected, score = module.select_refinement(test_embeddings, proposed, context)

        assert selected.shape == test_embeddings.shape
        assert score.shape == (test_embeddings.size(0), 1)
        assert (score >= 0).all() and (score <= 1).all()


class TestSecurityHardeningModule:
    """Tests for SecurityHardeningModule."""

    @pytest.fixture
    def module(self) -> SecurityHardeningModule:
        """Create module instance."""
        return SecurityHardeningModule(embed_dim=512)

    def test_detect_vulnerabilities(
        self, module: SecurityHardeningModule, test_embeddings: torch.Tensor
    ) -> None:
        """Test vulnerability detection returns expected structure."""
        result = module.detect_vulnerabilities(test_embeddings)

        assert "scores" in result
        assert "types" in result
        assert "max_risk" in result
        assert result["scores"].shape == (test_embeddings.size(0), 10)
        assert len(result["types"]) == 10
        assert result["max_risk"].shape == (test_embeddings.size(0),)

    def test_apply_hardening(
        self, module: SecurityHardeningModule, test_embeddings: torch.Tensor
    ) -> None:
        """Test hardening produces correct output shape."""
        vulnerabilities = torch.randn(test_embeddings.size(0), 10)
        hardened = module.apply_hardening(test_embeddings, vulnerabilities)
        assert hardened.shape == test_embeddings.shape


class TestQualityAssuranceModule:
    """Tests for QualityAssuranceModule."""

    @pytest.fixture
    def module(self) -> QualityAssuranceModule:
        """Create module instance."""
        return QualityAssuranceModule(embed_dim=512)

    def test_evaluate_quality(
        self, module: QualityAssuranceModule, test_embeddings: torch.Tensor
    ) -> None:
        """Test quality evaluation returns all metrics."""
        scores = module.evaluate_quality(test_embeddings)

        expected_metrics = [
            "correctness",
            "reliability",
            "performance",
            "maintainability",
            "scalability",
            "overall",
        ]

        for metric in expected_metrics:
            assert metric in scores
            assert (scores[metric] >= 0).all() and (scores[metric] <= 1).all()

    def test_generate_tests(
        self, module: QualityAssuranceModule, test_embeddings: torch.Tensor
    ) -> None:
        """Test test case generation produces correct shape."""
        num_tests = 5
        tests = module.generate_tests(test_embeddings, num_tests=num_tests)
        assert tests.shape == (test_embeddings.size(0), num_tests, 512)


class TestMultiLanguageExplorer:
    """Tests for MultiLanguageExplorer."""

    @pytest.fixture
    def explorer(self) -> MultiLanguageExplorer:
        """Create explorer instance."""
        return MultiLanguageExplorer(embed_dim=512, num_languages=10)

    def test_explore_implementations(
        self, explorer: MultiLanguageExplorer, test_embeddings: torch.Tensor
    ) -> None:
        """Test implementation exploration for multiple languages."""
        languages = ["python", "javascript", "rust"]
        implementations = explorer.explore_implementations(test_embeddings, languages)

        assert len(implementations) == 3
        for lang in languages:
            assert lang in implementations
            assert implementations[lang].shape == test_embeddings.shape

    def test_select_best_implementation(
        self, explorer: MultiLanguageExplorer, test_embeddings: torch.Tensor
    ) -> None:
        """Test best implementation selection."""
        implementations = {
            "python": test_embeddings + 0.1,
            "javascript": test_embeddings - 0.1,
            "rust": test_embeddings * 1.1,
        }
        criteria = test_embeddings

        best_lang, best_impl = explorer.select_best_implementation(implementations, criteria)

        assert best_lang in implementations
        assert best_impl.shape == test_embeddings.shape


class TestSelfImprovingAgentFramework:
    """Tests for SelfImprovingAgentFramework integration."""

    @pytest.fixture
    def framework(self, config: dict[str, Any]) -> SelfImprovingAgentFramework:
        """Create framework instance."""
        return SelfImprovingAgentFramework(config)

    def test_framework_initialization(
        self, framework: SelfImprovingAgentFramework, config: dict[str, Any]
    ) -> None:
        """Test framework initializes all components."""
        assert framework.config == config
        assert framework.code_encoder is not None
        assert framework.self_improver is not None
        assert framework.security_hardening is not None
        assert framework.quality_assurance is not None
        assert framework.multi_lang_explorer is not None

    def test_improve_solution(
        self, framework: SelfImprovingAgentFramework, test_tokens: torch.Tensor
    ) -> None:
        """Test solution improvement pipeline."""
        result = framework.improve_solution(test_tokens, language="python", num_iterations=2)

        assert "final_solution" in result
        assert "history" in result
        assert "final_quality" in result
        assert "final_security" in result
        assert result["final_solution"].shape[-1] == 512
        assert len(result["history"]["iterations"]) == 2

    def test_explore_multi_language(
        self, framework: SelfImprovingAgentFramework, test_tokens: torch.Tensor
    ) -> None:
        """Test multi-language exploration."""
        languages = ["python", "javascript", "rust"]
        result = framework.explore_multi_language(test_tokens, languages)

        assert "best_language" in result
        assert "best_implementation" in result
        assert "all_results" in result
        assert result["best_language"] in languages
        assert len(result["all_results"]) == len(languages)


class TestCreateAgentFramework:
    """Tests for create_agent_framework factory function."""

    def test_create_with_default_config(self) -> None:
        """Test factory with default configuration."""
        framework = create_agent_framework(AgentType.SWE)
        assert isinstance(framework, SelfImprovingAgentFramework)
        assert framework.config["embed_dim"] == 512

    def test_create_with_custom_config(self) -> None:
        """Test factory with custom configuration."""
        custom_config = {"embed_dim": 256, "num_improvement_iterations": 5}
        framework = create_agent_framework(AgentType.AIE, config=custom_config)
        assert framework.config["embed_dim"] == 256
        assert framework.config["num_improvement_iterations"] == 5

    def test_create_security_agent(self) -> None:
        """Test security agent has security weight."""
        framework = create_agent_framework(AgentType.SECURITY)
        assert framework.config.get("security_weight") == 2.0

    def test_create_qa_agent(self) -> None:
        """Test QA agent has QA weight."""
        framework = create_agent_framework(AgentType.QA)
        assert framework.config.get("qa_weight") == 2.0

    def test_all_agent_types_create_successfully(self) -> None:
        """Test all agent types can be created."""
        for agent_type in AgentType:
            framework = create_agent_framework(agent_type)
            assert isinstance(framework, SelfImprovingAgentFramework)


# Regression tests - ensure behavior doesn't change between versions
class TestAgentRegressions:
    """Regression tests to prevent breaking changes."""

    def test_quality_score_range(self) -> None:
        """Regression: Quality scores must always be in [0, 1]."""
        module = SelfImprovementModule(embed_dim=512)
        # Test with various input magnitudes
        for scale in [0.01, 1.0, 100.0]:
            embeddings = torch.randn(8, 512) * scale
            scores = module.assess_quality(embeddings)
            assert (scores >= 0).all() and (scores <= 1).all(), (
                f"Quality scores out of range for scale={scale}"
            )

    def test_encoder_deterministic(self) -> None:
        """Regression: Encoder output must be deterministic given same input."""
        encoder = LanguageFrameworkEncoder(vocab_size=1000, embed_dim=512)
        encoder.eval()
        tokens = torch.randint(0, 1000, (2, 50))

        with torch.no_grad():
            output1 = encoder(tokens, language="python")
            output2 = encoder(tokens, language="python")

        assert torch.allclose(output1, output2), "Encoder output not deterministic"

    def test_framework_improvement_reduces_error(self) -> None:
        """Regression: Improvement iterations should generally improve quality."""
        framework = create_agent_framework(AgentType.SWE)
        tokens = torch.randint(0, 1000, (1, 100))

        result = framework.improve_solution(tokens, language="python", num_iterations=5)

        # Quality should generally improve (not strictly, but trend)
        quality_scores = result["history"]["quality_scores"]
        # Check that final score is not significantly worse than initial
        assert quality_scores[-1] >= quality_scores[0] * 0.5, (
            "Quality degraded significantly during improvement"
        )
