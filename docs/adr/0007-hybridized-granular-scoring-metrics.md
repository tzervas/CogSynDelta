# ADR-0007: Hybridized Granular Scoring Metrics for Learning Optimization

**Status**: Proposed
**Date**: 2026-01-18
**Decision Makers**: @tzervas
**Technical Story**: Enhance scoring accuracy for self-improving agents and memory management

## Context

The current CogSynDelta system relies on simple mean-based aggregation for scoring metrics across various components (quality assessment, security evaluation, improvement tracking, and multi-language exploration). This approach, while functional, lacks the granularity needed for sophisticated learning and memory management decisions.

### Current Limitations

1. **Oversimplified Aggregation**: Using `.mean().item()` on batched tensors loses important statistical information about metric distributions, variance, and outliers.

2. **Lack of Temporal Context**: Current metrics don't account for trends, volatility, or historical patterns that could inform optimization decisions.

3. **Uniform Weighting**: All metrics are treated equally regardless of their reliability, stability, or predictive power.

4. **Memory Inefficiency**: Raw metric storage without compaction leads to unnecessary memory overhead as history accumulates.

5. **Cold Start Problems**: Early system operation lacks sufficient historical data for robust decision-making.

### Business and Technical Requirements

- **Accuracy**: More precise scoring enables better optimization decisions
- **Efficiency**: Memory-conscious storage and computation
- **Adaptability**: System should improve as historical data accumulates
- **Robustness**: Handle edge cases and noisy data gracefully
- **Scalability**: Performance should not degrade with increased history

## Decision

We will implement a **Hybridized Granular Scoring System (HGSS)** that combines multiple statistical measures, temporal analysis, and adaptive weighting to provide more accurate scoring for learning and memory management decisions.

### Core Components

1. **Multi-Statistic Aggregation Layer**
2. **Temporal Smoothing and Trend Analysis**
3. **Adaptive Weighting Engine**
4. **Differential Compression for History**
5. **Cold Start Fallback Mechanisms**

### Key Features

- **Granular Statistics**: Mean, median, variance, skewness, kurtosis, percentiles
- **Temporal Analysis**: Exponential moving averages, trend detection, volatility measures
- **Quality Enrichment**: Confidence intervals, reliability scores, outlier detection
- **Memory Optimization**: Differential storage, compression, selective retention
- **Adaptive Behavior**: Transition from calculable metrics to historical patterns

## Rationale

### Why This Approach

The hybridized approach addresses the fundamental limitation of mean-based scoring by providing richer statistical context while maintaining computational efficiency. By combining multiple metrics with temporal analysis, the system can make more informed decisions about learning direction and memory management.

#### Statistical Granularity Benefits
- **Variance Awareness**: High variance indicates unreliable metrics, low confidence decisions
- **Distribution Shape**: Skewness/kurtosis reveal data characteristics for better modeling
- **Outlier Robustness**: Percentiles provide robust central tendency measures

#### Temporal Intelligence Benefits
- **Trend Detection**: Identify improving vs. deteriorating performance patterns
- **Volatility Assessment**: Distinguish stable improvements from noisy fluctuations
- **Momentum Analysis**: Recognize acceleration/deceleration in optimization progress

#### Adaptive Weighting Benefits
- **Reliability-Based**: More weight on stable, consistent metrics
- **Context-Aware**: Adjust based on available historical data
- **Performance-Driven**: Learn which metrics correlate with successful outcomes

### Alternatives Considered

#### Option 1: Simple Mean with Confidence Intervals

- **Pros**: Straightforward implementation, maintains backward compatibility
- **Cons**: Still loses distribution information, doesn't address temporal aspects
- **Why Rejected**: Insufficient granularity for complex optimization decisions

#### Option 2: Full Statistical Suite from Day One

- **Pros**: Maximum information capture, comprehensive analysis
- **Cons**: High computational overhead, memory intensive, complex cold start
- **Why Rejected**: Overkill for early system operation, potential performance bottlenecks

#### Option 3: Machine Learning-Based Scoring

- **Pros**: Can learn complex patterns, adaptive to system behavior
- **Cons**: Requires training data, black-box decision making, computational complexity
- **Why Rejected**: Premature optimization, adds unnecessary complexity before basic metrics are well-understood

#### Option 4: External Analytics Service

- **Pros**: Offloads computation, potentially more sophisticated analysis
- **Cons**: Network dependency, latency, data privacy concerns, vendor lock-in
- **Why Rejected**: Core system should be self-contained, external dependencies reduce reliability

## Consequences

### Positive

- **Improved Optimization Accuracy**: Better decisions lead to faster convergence and higher quality outcomes
- **Enhanced Memory Management**: Smarter retention policies reduce memory footprint while preserving important information
- **Adaptive System Behavior**: System naturally improves decision-making as it accumulates experience
- **Robustness to Noise**: Statistical measures provide resilience against measurement errors and outliers
- **Research Enablement**: Richer metrics support advanced analysis and algorithm development

### Negative

- **Increased Complexity**: More sophisticated algorithms require careful implementation and testing
- **Computational Overhead**: Additional calculations may impact real-time performance
- **Memory Trade-offs**: Storing additional statistics increases short-term memory usage
- **Cold Start Period**: Initial performance may be less optimal until sufficient history accumulates

### Neutral

- **API Changes**: Existing `.mean().item()` calls will need updates to use new HGSS interface
- **Storage Format Evolution**: Historical data may need migration or versioning
- **Monitoring Requirements**: Additional metrics will require new monitoring and alerting

## Implementation

### Phase 1: Core HGSS Framework

1. **Create HGSS Module** (`src/cogsyndelta/scoring/hgss.py`)
   - Define `GranularScorer` class with statistical computation methods
   - Implement temporal analysis functions
   - Add adaptive weighting logic

2. **Statistical Computation Layer**
   ```python
   class GranularScorer:
       def compute_statistics(self, values: torch.Tensor) -> dict[str, float]:
           """Compute comprehensive statistics for scoring."""
           return {
               'mean': values.mean().item(),
               'median': values.median().item(),
               'std': values.std().item(),
               'variance': values.var().item(),
               'skewness': self._compute_skewness(values),
               'kurtosis': self._compute_kurtosis(values),
               'p25': torch.quantile(values, 0.25).item(),
               'p75': torch.quantile(values, 0.75).item(),
               'iqr': torch.quantile(values, 0.75).item() - torch.quantile(values, 0.25).item(),
               'robust_mean': self._robust_mean(values),
           }
   ```

3. **Temporal Analysis Layer**
   ```python
   def compute_temporal_metrics(self, history: list[float], window: int = 10) -> dict[str, float]:
       """Analyze temporal patterns in metric history."""
       if len(history) < window:
           return self._cold_start_fallback(history)

       recent = torch.tensor(history[-window:])
       return {
           'ema': self._exponential_moving_average(history),
           'trend': self._compute_trend(history),
           'volatility': recent.std().item(),
           'momentum': self._compute_momentum(history),
           'stability': self._compute_stability(history),
       }
   ```

### Phase 2: Integration Points

1. **Agent Framework Integration**
   - Replace `.mean().item()` calls in `SelfImprovingAgentFramework`
   - Update `improve_solution` and `explore_multi_language` methods
   - Add HGSS scoring to quality assessment pipeline

2. **Memory Management Integration**
   - Enhance tiered memory system with HGSS-informed retention decisions
   - Implement differential compression for metric history
   - Add selective compaction based on statistical significance

3. **Quality Assurance Integration**
   - Update `QualityAssuranceModule` to use HGSS for comprehensive evaluation
   - Enhance test generation with statistical quality targets

### Phase 3: Adaptive Weighting System

1. **Weight Learning Algorithm**
   ```python
   def compute_adaptive_weights(self, statistics: dict, temporal: dict, history_length: int) -> dict[str, float]:
       """Compute adaptive weights based on data characteristics."""
       # Early stage: favor calculable, stable metrics
       if history_length < 10:
           return self._early_stage_weights(statistics)

       # Mature stage: incorporate temporal patterns
       reliability_weights = self._compute_reliability_weights(statistics, temporal)
       predictive_weights = self._compute_predictive_weights(statistics, temporal)

       return self._combine_weights(reliability_weights, predictive_weights)
   ```

2. **Cold Start Handling**
   - **Phase 1 (< 5 samples)**: Pure statistical measures with equal weighting
   - **Phase 2 (5-20 samples)**: Introduce basic temporal smoothing
   - **Phase 3 (20+ samples)**: Full HGSS with learned weights

### Phase 4: Memory Optimization

1. **Differential Storage**
   ```python
   def compress_history(self, history: list[dict]) -> bytes:
       """Compress historical metrics using differential encoding."""
       if len(history) < 2:
           return self._serialize(history)

       # Compute differences from baseline
       baseline = history[0]
       differentials = []
       for entry in history[1:]:
           diff = self._compute_differential(baseline, entry)
           differentials.append(diff)

       return self._compress_differentials(baseline, differentials)
   ```

2. **Selective Retention**
   - Retain high-variance periods for trend analysis
   - Compress stable periods with lossy compression
   - Prioritize recent history over distant past

### Phase 5: Monitoring and Validation

1. **Metric Quality Assessment**
   - Track HGSS computation performance
   - Monitor scoring accuracy vs. simple mean
   - Validate statistical assumptions

2. **System Health Checks**
   - Ensure HGSS doesn't degrade real-time performance
   - Monitor memory usage patterns
   - Validate optimization improvements

## Testing Strategy

### Unit Tests
- Statistical computation accuracy
- Temporal analysis correctness
- Adaptive weighting behavior
- Memory compression/decompression

### Integration Tests
- End-to-end agent improvement with HGSS
- Memory management with statistical retention
- Cold start behavior validation

### Performance Benchmarks
- Computation time comparison (HGSS vs. mean)
- Memory usage with/without compression
- Optimization convergence rates

### Validation Metrics
- Scoring accuracy improvement
- False positive/negative rates in decisions
- Memory efficiency gains
- System stability under various conditions

## Migration Strategy

### Backward Compatibility
- Maintain existing API during transition
- Add HGSS as optional enhancement
- Gradual rollout with feature flags

### Data Migration
- Convert existing mean-based history to HGSS format
- Implement versioning for stored metrics
- Provide migration utilities

### Rollback Plan
- Ability to disable HGSS and revert to mean-based scoring
- Performance monitoring with automatic fallback
- Gradual feature activation

## Success Criteria

1. **Accuracy Improvement**: HGSS scoring shows >15% better decision accuracy than mean-based
2. **Performance**: <5% computational overhead compared to mean-based scoring
3. **Memory Efficiency**: >30% reduction in historical metric storage
4. **Adaptability**: System performance improves as history accumulates
5. **Robustness**: No degradation in edge cases or noisy data conditions

## Future Considerations

### Advanced Features
- **Machine Learning Integration**: Use HGSS features to train predictive models
- **Multi-Objective Optimization**: Handle conflicting optimization goals
- **Contextual Adaptation**: Adjust scoring based on task characteristics
- **Federated Learning**: Share statistical insights across agent instances

### Research Opportunities
- **Optimal Statistic Combinations**: Research which statistical measures provide most value
- **Temporal Window Optimization**: Determine ideal history windows for different metrics
- **Compression Algorithms**: Explore advanced compression for metric history
- **Uncertainty Quantification**: Better handling of scoring confidence

## References

- [ADR-0002: Tiered Memory Architecture](0002-tiered-memory-architecture.md)
- [ADR-0004: Graceful Degradation Patterns](0004-graceful-degradation-patterns.md)
- [PyTorch Statistical Functions](https://pytorch.org/docs/stable/torch.html#statistical-functions)
- [Exponential Moving Average](https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average)
- [Robust Statistics](https://en.wikipedia.org/wiki/Robust_statistics)

---

*This ADR proposes a comprehensive enhancement to CogSynDelta's scoring system, balancing accuracy, efficiency, and adaptability for advanced learning optimization.*
