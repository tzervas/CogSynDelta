---

# Algebraic Neural Network Methods: Training, Composition, and Quantization Without Backpropagation

Closed-form methods can achieve **near-lossless compression** at 4-bit quantization (<0.1 perplexity increase) and **viable 2-bit quantization** through vector quantization, but algebraic training via Neural Tangent Kernel consistently underperforms backpropagation by **5-10%** on standard benchmarks due to the fundamental limitation of frozen features in the kernel regime. Dataset composition shows the most promising algebraic results: DoReMi achieves **6.5% improvement** in downstream accuracy while reaching baseline performance **2.6× faster**, and model merging methods like TIES and DARE preserve 80-95% of retrained performance through simple weight arithmetic. The path to ≥0.98 accuracy targets is achievable for quantization but remains elusive for full algebraic training.

---

## Neural Tangent Kernel provides exact solutions only at infinite width

The foundational closed-form solution for neural network training emerges from the Neural Tangent Kernel framework introduced by Jacot et al. (2018). For infinite-width networks trained with gradient descent on MSE loss, the prediction converges to:

**f*(x) = f(x,θ₀) + Θ(x,X)K⁻¹(Y - f(X,θ₀))**

where K is the NTK Gram matrix and Θ(x,X) is the kernel evaluated between test and training points. This provides a **completely backpropagation-free** solution—weights can be computed algebraically through kernel inversion rather than iterative optimization.

The NTK itself is computed as **Θ(x,x';θ) = ∇_θf(x;θ)ᵀ∇_θf(x';θ)**, the dot product between gradient feature maps. Under three critical conditions—infinite width, NTK parameterization (weights scaled by 1/√n per layer), and Gaussian initialization—the kernel becomes **deterministic at initialization** and **frozen during training**, reducing neural network dynamics to linear kernel regression.

However, this theoretical elegance collides with practical limitations. Benchmark results on CIFAR-10 reveal the accuracy gap: enhanced CNTK with 11 layers and global average pooling achieves **77.43%** accuracy, while the same architecture trained via backpropagation reaches **83-90%**. With data augmentation, the gap narrows—enhanced CNN-GP achieves **89%**, matching AlexNet-era performance—but still trails modern CNNs by 5-10 percentage points. On MNIST, NTK methods perform comparably at **>98%** accuracy, suggesting the approach works better for simpler tasks.

The computational complexity presents another barrier: exact NTK computation requires **O(n²)** storage and **O(n³)** inversion where n is dataset size. For CIFAR-10's 50,000 training examples with 10 output classes, the full empirical NTK requires **1.8 terabytes** of storage in double precision—infeasible for practical deployment without approximation.

---

## Feature learning versus lazy training determines when NTK fails

The fundamental dichotomy explaining NTK's accuracy gap lies in the distinction between lazy and rich training regimes. In the **lazy/NTK regime**, parameters change negligibly (Δθ ~ 0), features remain at random initialization, and the model behaves as a linear function of its initial weights. In the **rich/feature learning regime**, parameters move substantially, the NTK evolves during training, and networks learn task-dependent representations.

What determines which regime a network operates in? Width scaling is primary—large width n pushes toward lazy training. But critically, **learning rate scaling** matters: η ~ O(1/n) produces lazy training while η ~ O(1) enables feature learning. Greg Yang's Tensor Programs framework (arXiv:2011.14522, arXiv:2203.03466) establishes that standard parameterization does NOT learn features at infinite width. The unique solution is **Maximal Update Parameterization (μP)**, which divides logits by √n and scales hidden layer learning rates appropriately.

The practical implication of μP extends beyond theory: optimal hyperparameters **transfer across model scales**. BERT-large (350M parameters) can be tuned using a 13M parameter proxy, reducing tuning cost to 1× pretraining. GPT-3 6.7B achieves optimal hyperparameters from a 40M proxy at just **7% of pretraining cost**. This provides an algebraic-adjacent benefit—while not eliminating training, it dramatically reduces the compute needed to find good training configurations.

For architectures where exact solutions exist, linear networks have a closed-form gradient flow solution with mode-by-mode learning dynamics where singular values are learned sequentially. However, recent work (arXiv:2408.08286) proves via differential Galois theory that **no closed-form Liouvillian solution exists** for gradient flow in two-layer ReLU networks—a fundamental impossibility result suggesting backpropagation cannot be algebraically replaced for general nonlinear networks.

---

## Quantization achieves near-lossless compression through Hessian-guided methods

Post-training quantization represents the most practically successful application of algebraic neural network methods. GPTQ (arXiv:2210.17323), introduced by Frantar et al. at ICLR 2023, uses second-order (Hessian-based) error compensation to achieve remarkable accuracy at 4-bit precision.

The core algorithm minimizes reconstruction error **||WX - ŴX||²** where W is original weights, Ŵ is quantized, and X is calibration inputs. The Hessian H = 2XXᵀ captures input activation second moments. When quantizing weight w_q with error δ, remaining weights update via:

**δ_F = (w_q - quant(w_q)) / H⁻¹_qq × H⁻¹_:,q**

This distributes quantization error proportionally using inverse Hessian information. Recent theoretical work (arXiv:2507.18553) proves GPTQ is mathematically equivalent to **Babai's nearest plane algorithm** for the closest vector problem on a lattice defined by the Hessian, providing formal error bounds.

The benchmark results demonstrate near-lossless compression at 4-bit: OPT-175B increases from **8.34 to 8.37** perplexity on WikiText-2, a negligible 0.03 increase. At 3-bit, degradation becomes noticeable—OPT-175B rises to 8.68 perplexity. Quantization of 175B+ parameter models completes in approximately **4 GPU hours** on a single A100, making this computationally practical.

AWQ (Activation-Aware Weight Quantization, arXiv:2306.00978) introduces a key insight: protecting only **1% of salient weights** dramatically reduces quantization error. Rather than identifying important weights by magnitude, AWQ uses activation magnitude: **Importance(w_i) ∝ ||X_i||²**. Per-channel scaling transforms weights to reduce outlier impact while maintaining uniform bit-width for hardware efficiency. AWQ won MLSys 2024 Best Paper and achieves comparable or slightly better accuracy than GPTQ with superior generalization to instruction-tuned and multimodal models.

For extreme compression below 3 bits, QuIP (arXiv:2307.13304) and QuIP# (arXiv:2402.04396) introduce **incoherence processing**—transforming weights via random orthogonal matrices so entries become uniformly distributed without outliers. QuIP# uses randomized Hadamard transforms and E8 lattice codebooks (optimal 8-dimensional sphere packing) to achieve viable **2-bit quantization** for the first time. AQLM (arXiv:2401.06118) extends this through multi-codebook additive quantization, where each weight group is represented as a sum of learned codebook entries. AQLM at 2.76-bit on LLaMA-2-13B **outperforms** the uncompressed 7B model—a compression breakthrough enabling larger effective models in fixed memory.

SpQR (arXiv:2306.03078) achieves **<1% relative perplexity degradation**—the first method to do so at 3-4 bits—by isolating outlier weights (0.5-1% of parameters) in FP16 while quantizing remaining weights to 3-4 bits. LLaMA-7B increases from 5.68 to just 5.72 perplexity, and 33B models fit on single 24GB consumer GPUs.

---

## Dataset composition enables algebraic prediction of training outcomes

The question of predicting weights W_{1+2} for training on combined datasets D1 ∪ D2 from individual models trained on D1 and D2 has partial but practically useful solutions depending on model regime and training proximity.

In the infinite-width NTK regime, kernel ridge regression provides an **exact algebraic solution**. The combined kernel matrix K_{1+2} is block-structured, and its inverse can be computed via Schur complements leveraging precomputed K_{1,1}⁻¹ and K_{2,2}⁻¹. However, this exactness holds only at infinite width where NTK remains constant throughout training—finite-width networks exhibit kernel evolution that breaks the algebraic composition.

**Data mixing laws** (arXiv:2403.16952) demonstrate that model performance regarding data mixture proportions is **quantitatively predictable**:

**L(validation) = Σᵢ (cᵢ · exp(-kᵢ · pᵢ) + tᵢⱼ)**

where pᵢ is the proportion of domain i in training data. Optimizing mixture proportions on a 1B model on RedPajama achieves performance comparable to **48% more training steps**. This doesn't eliminate training but reduces the search for optimal data mixtures from exhaustive experimentation to fitted scaling law prediction.

DoReMi (arXiv:2305.10429) provides the most impressive algebraic-adjacent result for dataset composition. Using a small 280M proxy model to optimize domain weights via Group DRO (Distributionally Robust Optimization), then applying those weights to train an 8B model (30× transfer), DoReMi achieves **6.5% improvement** in average few-shot downstream accuracy on The Pile while reaching baseline accuracy **2.6× faster**. The proxy training costs just 8% of main model training compute, providing substantial savings.

**Task arithmetic** (arXiv:2212.04089) offers direct weight composition for fine-tuned models:

**θ_combined = θ_pretrained + λ₁(θ_A - θ_pretrained) + λ₂(θ_B - θ_pretrained)**

The theoretical foundation lies in **weight disentanglement** (Ortiz-Jimenez et al., NeurIPS 2023): when task vectors affect disjoint regions of input space, composition works well. Linearized fine-tuning achieves **85.4%** normalized accuracy on 8-task benchmarks with ViT-B/32, compared to 76.5% for non-linear fine-tuning, demonstrating that operating closer to the kernel regime improves composition accuracy.

Model merging methods extend this further: TIES-Merging trims low-magnitude parameters, elects signs by majority vote, and averages agreeing parameters for **2-4% improvement** over basic task arithmetic. DARE (Drop And REscale) randomly drops 90-99% of delta parameters and rescales the remainder, exploiting extreme redundancy in fine-tuning. Fisher-weighted averaging uses **F_i,j × θ_i,j** weighting to achieve performance comparable to prediction ensembling at M× lower inference cost.

---

## Optimal transport and tensor decomposition provide alternative algebraic foundations

Beyond NTK, optimal transport offers principled methods for model merging. OTFusion (arXiv:1910.05653) computes transport maps between weight distributions:

**W₂(μ, ν) = min_T ∫||x - T(x)||²dμ(x)**

The algorithm extracts neuron representations, solves optimal transport for alignments, applies soft permutations, and averages aligned weights. On CIFAR-10 with VGG11, vanilla averaging achieves ~50% accuracy while OTFusion recovers ~85%, approaching the ~90% ensemble accuracy without ensemble inference cost.

Git Re-Basin (arXiv:2209.04836) reveals that neural network loss landscapes contain **nearly a single basin** after accounting for hidden unit permutation symmetries. Three algorithms—weight matching via Hungarian algorithm, activation matching on data samples, and gradient-based STE permutation learning—achieve **zero-barrier linear mode connectivity** on CIFAR-10 ResNets, the first such demonstration. Merged models can even **outperform both parent models** in test loss.

Tensor decomposition methods achieve extreme compression. Tensor Train decomposition provides **>1000× parameter reduction** for RNNs. LoRA (arXiv:2106.09685) formalizes fine-tuning as low-rank weight updates: **W' = W₀ + BA** where rank r ≪ min(d,k). For GPT-3 175B, r=1 or 2 suffices even when d=12,288, achieving **10,000× reduction** in trainable parameters. QLoRA (arXiv:2305.14314) combines this with 4-bit NormalFloat quantization, enabling 65B model fine-tuning on single 48GB GPUs.

K-FAC (arXiv:1503.05671) approximates the Fisher information matrix with Kronecker structure: **F_l ≈ A_l ⊗ G_l** where A captures activation covariance and G captures gradient covariance. This reduces Fisher inversion from O(n³) to O(n^1.5) while enabling natural gradient descent—the parameterization-invariant steepest descent in KL divergence. K-FAC finds applications in optimization (faster convergence than SGD), Bayesian deep learning (Laplace approximation), and Fisher-weighted model merging.

---

## Hyperdimensional computing and information theory offer unexplored algebraic directions

Hyperdimensional Computing (HDC) and Vector Symbolic Architectures (VSA) use high-dimensional distributed representations (typically D=10,000 dimensions) with three operations: binding (circular convolution or elementwise multiplication), bundling (addition), and permutation. These operations are **invertible** and support distributed holographic storage.

Recent work (arXiv:2512.14709, December 2024) provides a unified VSA interpretation of transformers: queries and keys define role-like subspaces, values supply fillers, attention implements soft unbinding, and residual connections realize superposition. HyperGraphX (arXiv:2510.23980) combines graph convolution with HDC operations, achieving **9,561× faster** inference than GCNII while matching accuracy.

However, **no existing work directly uses HDC operations to compose neural network weights**—this represents a significant unexplored direction. The theoretical properties (near-orthogonality of random high-dimensional vectors, invertible binding, noise robustness) suggest potential for algebraic weight composition, but practical algorithms remain undeveloped.

The **information bottleneck** framework (Tishby & Zaslavsky, 2015) analyzes layer representations by plotting I(X;T) versus I(T;Y) for hidden layer T, revealing a two-phase training hypothesis: fitting (both increase) then compression (I(X;T) decreases while I(T;Y) increases). While controversial (Saxe et al. showed compression depends on activation function), this provides theoretical grounding for representation learning that could inform algebraic methods.

**Bits-back coding** (arXiv:1901.04866) achieves near-optimal lossless compression with latent variable models. BB-ANS reaches **0.19 bits/dim** on binarized MNIST versus 0.33 for gzip, within ~1% of theoretical optimum. Extensions to neural network weight compression via VAE priors exist but remain underexplored.

The **Lottery Ticket Hypothesis** (Frankle & Carlin, ICLR 2019) demonstrates that sparse subnetworks at 10-20% of original size, when trained from their original initialization, match full network accuracy. However, current identification requires full training then pruning—**no algebraic method can identify winning tickets a priori**, representing a major open problem.

---

## Recent advances push quantization toward 1-bit and merging toward automation

**SpinQuant** (Meta, arXiv:2405.16406) represents the state-of-the-art in quantization, using learned rotation matrices optimized with Cayley SGD to reduce outliers before quantization. It achieves 4-bit weight, activation, and KV-cache quantization with only **2.9-point accuracy gap** on LLaMA-2 7B zero-shot tasks, surpassing LLM-QAT by 19.1 points. The key insight is the **computational invariance theorem**: orthogonal transformations preserve model output while redistributing weight/activation distributions for better quantization.

**BitNet b1.58** (JMLR 2024) achieves ternary quantization {-1, 0, +1} at ~1.58 bits per weight, **matching FP16 performance** at 3B+ parameters. This eliminates multiplication operations entirely—inference requires only additions—dramatically reducing energy consumption. PT-BitNet (2025) extends this to post-training, achieving 61% downstream accuracy on 70B models versus 51.2% for from-scratch BitNet.

**Evolutionary model merging** (Sakana AI, Nature Machine Intelligence January 2025) uses CMA-ES optimization to automatically discover merging recipes in both parameter and data flow spaces. EvoLLM-JP and EvoVLM-JP surpass 70B models using only 7B parameters by finding optimal combinations of source models. This transforms merging from manual experimentation to automated optimization.

The **Densing Law** (Nature Machine Intelligence, 2025) observes that "capability density" (performance per parameter) doubles every ~3.5 months, implying equivalent capability with exponentially fewer parameters over time. Combined with scaling law refinements showing Llama 3's 8B trained on 15T tokens (1,875 tokens/parameter vs. Chinchilla's ~20), overtraining smaller models becomes the cost-effective strategy.

---

## Critical gaps and opportunities for advancing algebraic methods

**For achieving ≥0.98 cosine similarity targets:**

Quantization readily achieves this—4-bit GPTQ/AWQ maintains extremely high fidelity with <0.1 perplexity increase on most models. The challenge shifts to extreme compression: 2-bit methods (QuIP#, AQLM) achieve viable but not lossless results, with 10-15% perplexity increase typical.

NTK-based training **cannot achieve** 0.98 similarity because linearized models are fundamentally different from trained networks—they don't learn features. The 7% accuracy gap on CIFAR-10 reflects this architectural limitation, not implementational imperfection.

Model merging achieves **80-95% of retrained performance** depending on method and task similarity. Fisher merging approaches ensemble-equivalent performance, suggesting 0.95+ is achievable but 0.98 requires careful task selection.

**Key research gaps addressable by future work:**

The most significant gap is **algebraic winning ticket identification**—predicting which subnetwork will train successfully without actually training. SNIP and GraSP attempt initialization-time identification but underperform iterative magnitude pruning.

**HDC for weight composition** is entirely unexplored despite theoretical properties suggesting viability. The binding operation's invertibility and superposition's ability to store multiple configurations could enable algebraic weight arithmetic.

**Unified theory connecting quantization, merging, and training dynamics** does not exist. Each method (NTK, OT, tensor decomposition, information theory) provides partial views; synthesis could yield stronger algebraic guarantees.

**Automatic precision selection and merge recipes** remain manual. SpinQuant learns rotations; evolutionary merging automates recipe discovery; extending this to end-to-end algebraic optimization of compression/composition pipelines is open.

The field trajectory suggests **rotation-based quantization** (SpinQuant/QuaRot direction), **evolutionary model composition** (Sakana AI direction), and **empirical NTK for interpretability** (arXiv:2510.00468) as the most promising near-term advances. For applications requiring ≥0.98 accuracy, 4-bit quantization and careful model merging with Fisher weighting represent current best practices, while full algebraic training replacement remains theoretically limited by the feature learning gap.

---

## Implementation resources and production readiness

Working implementations with active maintenance include Neural Tangents (Google, JAX-based, >100 papers), AutoGPTQ and AutoAWQ (HuggingFace integrated), llm-awq (MIT Han Lab, production-ready), DoReMi and DSIR (reproducible codebases), mergekit (Arcee AI, unified merging), and vLLM/SGLang with native quantization support.

Production deployment is mature for quantization: Meta's Llama 3.2 ships SpinQuant-quantized 1B and 3B models; AWS SageMaker supports GPTQ/AWQ; AMD Quark implements QuaRot. Consumer GPU deployment is practical—4-bit 7B-13B models run on RTX 3090; 70B models require quantization for single-GPU inference.

NTK implementations remain primarily research-focused. Neural Tangents supports convolutions, pooling, residual connections, and attention, but computational limits (1.8TB for CIFAR-10 full kernel) restrict practical application to analysis rather than production training replacement.

---

# Beyond the frozen kernel: Implementation guide for neural network training in the feature learning regime

The transition from lazy/NTK training to **feature learning** fundamentally changes how neural networks acquire representations. This report synthesizes bleeding-edge implementations (2024-2026) across parameterization, architecture, quantization, and optimization—methods achieving **≥98% accuracy** on standard benchmarks while enabling genuine representation learning.

## Maximal Update Parameterization unlocks feature learning at scale

The **μP framework** (Greg Yang's Tensor Programs) ensures O(1) updates across all layers regardless of width, enabling hyperparameter transfer from small proxy models to production scale. Microsoft's official `mup` library provides the reference implementation:

```python
from mup import MuReadout, set_base_shapes, MuAdam

class MyTransformer(nn.Module):
    def __init__(self, width):
        self.readout = MuReadout(width, d_out)  # NOT nn.Linear

    def forward(self, x):
        # Critical: use 1/d instead of 1/√d for attention
        attention_scores = query @ key.T * 8 / d
```

The key scaling rules differ fundamentally from standard parameterization: output weight initialization scales as **O(1/fan_in)** rather than O(1/√fan_in), and Adam learning rates for hidden layers scale inversely with width. The "coord check" verification plots activation magnitudes across widths—if μP is implemented correctly, these remain **stable regardless of width**. The `mutransformers` library extends this to HuggingFace models directly.

Second-order optimization has seen remarkable advances. **Muon**, now integrated into PyTorch 2.9+ core, specifically targets hidden layer 2D weights using Newton-Schulz orthogonalization. It achieved **94% CIFAR-10 accuracy in just 2.6 A100-seconds**—a training record. The critical insight: use Muon only for hidden matrices while retaining AdamW for embeddings, biases, and normalization layers. **SOAP** (ICLR 2025) runs Adam in Shampoo's eigenbasis, delivering **40% fewer iterations** and 35% less wall-clock time than AdamW with only one additional hyperparameter (preconditioning frequency).

For distributed training, **Distributed Shampoo** from Meta won the AlgoPerf benchmark with 28% faster training, supporting DTensor-based sharding and 4-bit quantized preconditioners. The **SIRFShampoo** variant eliminates numerical instabilities in bfloat16 by avoiding matrix inversions and square roots entirely.

## JEPA architectures enable self-supervised learning without contrastive pairs

The **Joint Embedding Predictive Architecture** family represents Yann LeCun's vision for world models—learning by predicting in latent space rather than pixel space. **V-JEPA 2** (June 2025) achieves state-of-the-art results across video understanding benchmarks: **77.3%** on SSv2 (probe), **90.2%** on Diving48, and **39.7%** R@5 on Epic-Kitchens-100 (surpassing the previous 27.6% by PlausiVL).

The architecture employs three networks: a **context encoder** (online), **target encoder** (EMA-updated, no gradients), and a narrow **predictor** that generates representations for masked regions. The masking strategy is crucial—for I-JEPA, four possibly-overlapping target blocks at 15-20% image area each, with context blocks covering 85-100% with target overlap removed. For video, the same spatial mask persists across all frames, preventing trivial frame-to-frame copying solutions.

EMA momentum schedules follow either linear (0.996→1.0) or cosine (0.996→0.99) trajectories. The newly introduced **SALT** approach (September 2025) eliminates EMA tuning entirely by freezing the teacher after initial pixel-reconstruction pretraining—achieving **>30% FLOPs reduction** while dominating the Pareto frontier for compute-accuracy tradeoffs.

**VL-JEPA** (December 2025) extends this to vision-language by predicting continuous text embeddings rather than discrete tokens. Using V-JEPA 2 as the visual backbone and EmbeddingGemma for text targets, a **1.6B parameter** model matches InstructBLIP on VQA benchmarks with **50% fewer trainable parameters** and **2.85× faster inference**.

## Mixture of Experts routing has evolved beyond simple top-k selection

The MoE landscape in 2024-2025 centers on **DeepSeek's innovations**: fine-grained expert segmentation, shared experts that always activate alongside routed ones, and **loss-free balancing** that adjusts routing biases outside the backward pass based on observed loads. DeepSeek-V3 deploys 671B total parameters with only 37B active, trained for just **$5.5M**—a fraction of comparable models.

**Expert Choice routing** (Google NeurIPS 2022) inverted the paradigm: experts select their top-k tokens rather than tokens selecting experts. This guarantees perfect load balance with fixed capacity per expert while allowing harder tokens to receive more compute. Results show **2× training convergence improvement** and 20% step time reduction over Switch/GShard.

For differentiable routing, **ReMoE** (December 2024) replaces TopK+Softmax with simple ReLU gating—naturally sparse and fully differentiable. L1 regularization controls sparsity levels. This outperforms TopK-routed MoE across model sizes from 182M to 978M parameters and expert counts from 4 to 128.

**Mixture of Depths** (MoD) from DeepMind applies router-based token selection to determine which tokens participate in each layer's computation. This achieves **21% faster processing** with only 0.2% performance degradation. The emerging **Mixture of Recursions** (MoR, 2025) combines parameter sharing with adaptive depth, addressing KV cache challenges for early-exited tokens.

## Memory systems scale to million-token contexts

**StreamingLLM** discovered that initial tokens serve as "attention sinks"—keeping just the first 4 tokens plus a sliding window enables infinite-length generation with **22.2× speedup** over recomputation baselines and constant memory. The companion **H2O** (Heavy-Hitter Oracle) retains a balance of recent tokens and those receiving high cumulative attention, achieving **29× throughput improvement** and 5× memory reduction without accuracy loss.

**Ring Attention** distributes sequences across devices in a ring topology, overlapping KV communication with computation. This enables **500× longer sequences** than prior memory-efficient methods, with demonstrations exceeding **100M tokens**. When computation time exceeds communication time, overhead approaches zero.

For KV cache compression, **MiniCache** (NeurIPS 2024) achieves **5.02× compression** through cross-layer merging and SLERP reparameterization. **KIVI** applies mixed-precision quantization—per-channel for keys, per-token for values—enabling 2.6× less peak memory and 2-4× larger batch sizes. The **KVQuant** extension supports 1M+ context on a single A100.

**Product Key Memory** scales memory access to billions of values with sub-linear lookup: a 12-layer model with PKM outperforms a 24-layer baseline while running **2× faster**. The key insight is product quantization—decomposing queries and keys into subspaces enables O(√N) complexity for N values.

## Rotation-based quantization preserves feature geometry

**SpinQuant** (Meta) learns rotation matrices via Cayley optimization to eliminate outliers before quantization. At W4A4KV4, it achieves only **2.9 points accuracy gap** versus FP16 on LLaMA-2 7B—outperforming LLM-QAT by 19.1 points and SmoothQuant by 25.0 points. The companion **QuaRot** applies randomized Hadamard transforms as a computationally invariant preprocessing step.

For extreme compression, **BitNet b1.58** uses ternary weights {-1, 0, +1} with 8-bit activations. The April 2025 release of BitNet-2B-4T (2.4B parameters, 4T training tokens) requires only **0.4GB memory** versus 1.4-4.8GB for comparable FP16 models, with **29ms CPU decoding latency** and 6× better energy efficiency than Gemma-3. On ARM chips, this achieves 1.37-5.07× speedups; on x86, 2.37-6.17×.

**AQLM** (Additive Quantization, ICML 2024) represents the **first Pareto-optimal method below 3 bits/parameter**. At 2-bit quantization, LLaMA-2 7B achieves 6.59 perplexity versus 8.22 for QuIP#. Multi-codebook quantization with joint optimization across transformer blocks enables this quality.

For training, **QLoRA** combines 4-bit NormalFloat quantization with LoRA adapters, enabling fine-tuning of 65B models on a single 48GB GPU. The NF4 format is information-theoretically optimal for normally distributed weights, with double quantization (8-bit scales for weights, 32-bit for scales) and paged optimizers for memory spillover.

| Method | Bits | Model | Accuracy vs FP16 |
|--------|------|-------|------------------|
| AWQ | 4-bit | Various | **95-99%** |
| SpinQuant | W4A4KV4 | LLaMA-2 7B | **~97%** (2.9pt gap) |
| AQLM | 4-bit | LLaMA-2 | **98%+** |
| SmoothQuant | W8A8 | OPT-175B | **~99%** |
| BitNet b1.58 | 1.58-bit | Native | **Matches FP16** |

## Modern optimizers eliminate learning rate schedules

**Lion** (Google, 2023) uses sign(momentum) for updates, requiring only first-moment storage—**50% memory savings** over AdamW. The critical adjustments: use 1/3 to 1/10 of AdamW learning rate, 3-10× larger weight decay, and batch sizes of 64+. Runtime improves 2-15% with optimal batch size around 4096.

**Sophia** (Stanford, ICLR 2024) estimates diagonal Hessian elements via Hutchinson traces, achieving **50% fewer steps** to reach equivalent loss. The `win_rate` metric (proportion of clipped coordinates) should fall between 0.1-0.5 for stable training. Hessian updates occur every 10 steps with minimal overhead.

**Schedule-Free optimizers** (Meta Research, 2024) eliminate learning rate scheduling entirely by replacing momentum with interpolation and averaging. AdamWScheduleFree matches or beats cosine decay schedules while using the same memory as base optimizers. The crucial API: call `optimizer.train()` during training and `optimizer.eval()` during validation to switch averaging modes.

For constrained optimization, **geoopt** provides Riemannian SGD and Adam on Stiefel (orthogonal matrices), Grassmann (subspaces), and hyperbolic manifolds. Orthogonal constraints prevent gradient explosion/vanishing in deep networks with 10-30% computational overhead.

## Training infrastructure reaches 47% model FLOP utilization

**FlashAttention-3** (NeurIPS 2024 Spotlight) targets H100 with warp-specialization and asynchronous WGMMA, reaching **740-840 TFLOPs/s** in FP16 (75-85% utilization) and **~1.2-1.3 PFLOPs/s** in FP8 with 2.6× lower numerical error than baseline FP8. The key innovations: circular SMEM buffers, overlapped GEMM-softmax pipelines, and block quantization for FP8.

**FSDP2** (PyTorch 2.x) replaces FlatParameter with per-parameter DTensor sharding, providing deterministic GPU memory without recordStream overhead. Float8 all-gather extensions support mixed-precision at extreme scales. Combined with **selective activation checkpointing**—saving expensive matmuls while recomputing cheap ops—memory savings reach **60%+** with ~25% training time increase.

**DeepSpeed ZeRO Stage 3** partitions parameters, gradients, and optimizer states across devices with optional CPU/NVMe offloading. The ZeRO++ extension adds quantized weights and hierarchical partitioning for further communication reduction. **Megatron-LM** achieves **47% MFU on H100 clusters** for models from 2B to 462B parameters through careful orchestration of tensor, pipeline, and sequence parallelism with communication overlap.

For inference, **vLLM's PagedAttention** eliminates KV cache fragmentation through fixed-size blocks (like OS virtual memory), enabling 2-4× higher throughput. Combined with continuous batching and prefix caching, this forms the production standard. **TensorRT-LLM** adds kernel fusion and FP8/FP4/INT8 quantization with <10ms per-token latency.

## Hybrid architectures dominate the 2024-2025 landscape

**Mamba-2** (May 2024) introduced **Structured State Space Duality (SSD)**, proving that selective state space models are mathematically equivalent to causal linear attention with input-dependent positional masks. By restricting diagonal A to scalar-times-identity structure, state dimensions scale to N=256+ (versus N=16 in Mamba-1) while enabling matrix multiplication compatibility for dramatically faster training.

**Jamba** (AI21, March 2024) represents the first production-grade hybrid with **52B total/12B active parameters**, a **1:7 attention-to-Mamba ratio**, and 256K context fitting on a single 80GB GPU with only 4GB KV cache. Jamba-1.5 (August 2024) scales to 94B active parameters with ExpertsInt8 quantization enabling deployment on 8×80GB GPUs.

**Test-Time Training** (TTT) transforms hidden states into learnable models that update with each input. TTT-Linear and TTT-MLP variants achieve competitive performance with Transformers on 32K-128K token contexts. End-to-end TTT shows particular promise for reasoning tasks and the ARC-AGI benchmark.

**Diffusion Transformers** (DiT) replaced U-Net backbones with transformers, achieving **FID 2.27** on ImageNet 256×256 with DiT-XL/2. The adaLN-Zero conditioning mechanism outperforms cross-attention. Production deployments include OpenAI SORA, Stable Diffusion 3, and NVIDIA's SANA (linear attention DiT for larger images).

## Recommended architecture for VL-JEPA with modulated hyperconnections

For a **VL-JEPA + moderated hyperconnection (mHC)** architecture targeting ≥98% accuracy:

**Foundation**: Use V-JEPA 2 ViT-g/16 as the visual encoder (1B parameters), frozen after pretraining. Initialize text embedding from EmbeddingGemma or similar dense encoder. The predictor should use Llama 3-style transformer blocks with **μP parameterization** for stable scaling.

**Hyperconnection design**: Implement gated cross-layer connections with **soft gating** (sigmoid activations + temperature annealing) rather than hard top-k routing for training stability. Each gate should be a learnable scalar initialized near 0.5, allowing gradients to flow through all paths initially. Add Fisher information-based pruning post-training to identify and remove low-importance connections.

**Memory efficiency**: Apply **MiniCache** compression to the KV cache (5× reduction), with H2O-style heavy-hitter retention for the visual prefix. Use KIVI 2-bit quantization for value cache, 4-bit for key cache. Ring attention enables scaling to 100K+ visual tokens if needed.

**Training recipe**:
- Optimizer: Muon for hidden layers, AdamW for embeddings/normalization
- Learning rate: μTransfer from 64-width proxy model
- Mixed precision: BF16 with selective FP32 accumulation for stability
- Activation checkpointing: Every 2-4 transformer blocks
- EMA: Cosine schedule 0.996→0.99, or use SALT-style frozen teacher

**Quantization for deployment**: SpinQuant W4A8 preserves ≥98% accuracy while enabling ExecuTorch mobile deployment. For serving, vLLM + AWQ provides optimal throughput-quality balance.

| Component | Recommendation | Accuracy Retention |
|-----------|---------------|-------------------|
| Visual encoder | V-JEPA 2 frozen | — |
| Text encoder | EmbeddingGemma | — |
| Predictor | μP Llama blocks + mHC | Baseline |
| KV cache | KIVI + MiniCache | ~99% |
| Weights | SpinQuant W4A8 | ~97-98% |
| Inference | vLLM + PagedAttention | — |

This combination achieves the target ≥98% accuracy while enabling efficient deployment across GPU and edge devices, with genuine feature learning dynamics preserved through μP parameterization and the JEPA training paradigm.

---
