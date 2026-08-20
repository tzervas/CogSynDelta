"""PoC-2: registry, second region, softmax router."""

from __future__ import annotations

import pytest
import torch

from cogsyndelta.contracts.config import RouteConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.contracts.metrics import MetricsStatus
from cogsyndelta.contracts.region import CognitiveRegion
from cogsyndelta.contracts.registry import RegionRegistry
from cogsyndelta.poc.regions import ResidualMLPRegion
from cogsyndelta.poc.route import default_two_region_mind, run_route
from cogsyndelta.poc.router import SoftmaxRouter
from cogsyndelta.poc.vae import LatentVAE


def test_regions_satisfy_protocol() -> None:
    mlp = ResidualMLPRegion(dim=16, hidden_dim=32)
    vae = LatentVAE(input_dim=16, hidden_dim=32, latent_dim=4)
    assert isinstance(mlp, CognitiveRegion)
    assert isinstance(vae, CognitiveRegion)


def test_latent_vae_activate_keeps_train_forward() -> None:
    torch.manual_seed(0)
    model = LatentVAE(input_dim=16, hidden_dim=32, latent_dim=4)
    x = torch.randn(3, 16)
    activated = model.activate(x)
    assert activated.shape == x.shape
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape
    assert mu.shape == (3, 4)
    assert logvar.shape == (3, 4)


def test_residual_mlp_preserves_shape() -> None:
    torch.manual_seed(1)
    region = ResidualMLPRegion(dim=8, hidden_dim=16)
    x = torch.randn(4, 8)
    y = region.activate(x)
    assert y.shape == x.shape
    assert not torch.equal(y, x)


def test_registry_register_get_duplicate() -> None:
    registry = RegionRegistry()
    a = ResidualMLPRegion(dim=8, name="residual_mlp")
    registry.register(a)
    assert registry.get("residual_mlp") is a
    assert registry.names() == ("residual_mlp",)
    assert len(registry) == 1
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(ResidualMLPRegion(dim=8, name="residual_mlp"))
    with pytest.raises(KeyError, match="unknown"):
        registry.get("missing")


def test_default_mind_has_two_same_dim_regions() -> None:
    registry = default_two_region_mind(stream_dim=16, hidden_dim=32, latent_dim=4)
    assert registry.names() == ("residual_mlp", "stream_vae")
    x = torch.randn(2, 16)
    for region in registry.regions():
        out = region.activate(x)
        assert out.shape == x.shape


def test_softmax_router_weights_sum_to_one() -> None:
    torch.manual_seed(2)
    registry = default_two_region_mind(8, 16, 4)
    router = SoftmaxRouter(dim=8, n_regions=2, top_k=1)
    stream = torch.randn(6, 8)
    result = router.route(stream, registry.regions())
    sums = result.weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert result.output.shape == stream.shape
    assert result.topk_index.shape == (6, 1)
    assert abs(sum(result.load.values()) - 1.0) < 1e-5


def test_softmax_router_topk1_is_one_hot_mix() -> None:
    torch.manual_seed(3)
    registry = default_two_region_mind(8, 16, 4)
    regions = registry.regions()
    router = SoftmaxRouter(dim=8, n_regions=2, top_k=1)
    stream = torch.randn(5, 8)
    result = router.route(stream, regions)
    # each token equals the winning region's activate (weight == 1 after renorm)
    for b in range(stream.size(0)):
        winner = int(result.topk_index[b, 0].item())
        expected = regions[winner].activate(stream[b : b + 1])
        assert torch.allclose(result.output[b : b + 1], expected, atol=1e-5)


def test_route_bench_records_pass() -> None:
    cfg = RouteConfig(stream_dim=32, hidden_dim=64, latent_dim=8, top_k=1, batch_size=8)
    ctx = DeviceContext.resolve("cpu")
    out = run_route(cfg, ctx, seed=11)
    assert out["route"]["regions"] == ["residual_mlp", "stream_vae"]
    assert out["route"]["gate"] == "softmax_topk"
    by_name = {r.name: r for r in out["records"]}
    assert by_name["softmax_router_weights"].status == MetricsStatus.PASS
    assert by_name["routed_stream_quant8"].status == MetricsStatus.PASS
    assert by_name["routed_stream_quant8"].measured["fidelity"] >= 0.90
