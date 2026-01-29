# Tripartite Empathy Decomposition in Large Language Models

**Final Technical Report**

*January 29, 2026*

---

## Executive Summary

This study investigates whether empathetic language representations in LLMs decompose into distinct subspaces corresponding to theoretical empathy subtypes: **Cognitive** (perspective-taking), **Affective** (emotional resonance), and **Instrumental** (problem-solving). Using a dual-path validation approach combining geometry-driven SAE discovery with theory-driven probe training, we analyzed 4 open-weight models (7-8B parameters).

### Key Findings

1. **Separation Confirmed but Non-Specific**: Mean cosine similarity between Cognitive and Affective empathy directions is **-0.29** across all models. However, **validation experiments revealed non-empathy controls show identical structure** (mean -0.49 for both conditions).

2. **Consistent Across Architectures**: All tested models show the same separation pattern, but this likely reflects general response distinctiveness rather than empathy-specific encoding.

3. **Weak SAE-Probe Convergence**: SAE clustering identifies k=2 clusters vs. theoretical k=3, with mean silhouette score of 0.28.

4. **Critical Validation Finding**: Control analysis shows the tripartite separation is **not unique to empathy**—any set of distinct response types produces similar geometric structure.

5. **Null Distribution Test**: Empathy cosines are **less separated than random label permutations** (Z=+13.9), definitively showing the "separation" is not meaningful.

6. **Steering Test**: Applying Cognitive-Affective direction vectors produces output changes, but these don't map clearly to empathy style shifts.

### Revised Conclusion

The tripartite separation is a methodological artifact, not empathy-specific encoding. The null distribution test is definitive: random label shuffling produces MORE separation than empathy labels. This is a critical negative result for representation engineering methodology.

---

## 1. Introduction

### 1.1 Research Question

Do language models represent empathy as a unified concept, or does it decompose into distinct representational subspaces corresponding to psychological theory?

### 1.2 Theoretical Background

Psychological literature distinguishes three empathy components:

- **Cognitive Empathy**: Understanding another's mental state ("I can see why you feel that way")
- **Affective Empathy**: Sharing emotional experience ("That must be really hard")
- **Instrumental Empathy**: Action-oriented support ("Here's what you could try")

If these are computationally distinct, we expect:
1. Linear probes can separate them (high AUROC)
2. Direction vectors have low cosine similarity (< 0.5)
3. Unsupervised clustering (SAEs) recovers similar structure

### 1.3 Hypotheses

- **H1 (Separation)**: cos(Cognitive, Affective) < 0.5
- **H2 (Convergence)**: SAE clusters align with probe-derived directions
- **H3 (Consistency)**: Pattern holds across model architectures

---

## 2. Methodology

### 2.1 Dataset

| Dataset | Description | Samples |
|---------|-------------|---------|
| `triplets_filtered.json` | 90 scenarios × 3 response types | 270 |
| `controls_non_empathy.json` | Non-empathetic emotional responses | 60 |
| `controls_valence_stripped.json` | Valence-neutral versions | 90 |
| **Total** | | **420** |

Each scenario includes matched Cognitive, Affective, and Instrumental responses generated via Claude API and human-filtered for quality.

### 2.2 Models Tested

| Model | Parameters | Architecture | Layer Analyzed |
|-------|------------|--------------|----------------|
| Qwen2.5-7B | 7B | Qwen2 | 14 |
| Mistral-7B | 7B | Mistral | 16 |
| Llama-3-8B | 8B | Llama 3 | 16 |
| Llama-3.1-8B | 8B | Llama 3.1 | 16 |

### 2.3 Dual-Path Validation

**Experiment A: Geometry-Driven (SAE)**
- Train sparse autoencoders on residual stream activations
- Apply k-means clustering to learned features
- Measure cluster purity and silhouette scores
- Compare optimal k to theoretical k=3

**Experiment B: Theory-Driven (Probes)**
- Train linear probes for each empathy subtype
- Extract direction vectors (probe weights)
- Compute pairwise cosine similarities
- Validate with held-out test set

**Convergence Analysis**
- Compare SAE cluster centroids to probe directions
- Measure alignment via cosine similarity
- Assess whether unsupervised and supervised methods agree

### 2.4 Infrastructure

- **Activation Extraction**: TransformerLens
- **SAE Training**: SAE-Lens (100 epochs)
- **Probe Training**: Logistic regression (50 epochs)
- **Compute**: RunPod GPUs (RTX A5000, A6000)
- **Total Runtime**: ~25 minutes per model

---

## 3. Results

### 3.1 Probe Separation (Experiment B)

| Model | cos(Cog,Aff) | cos(Cog,Instr) | cos(Aff,Instr) |
|-------|--------------|----------------|----------------|
| Qwen2.5-7B | **-0.322** | -0.287 | -0.361 |
| Llama-3.1-8B | **-0.317** | -0.360 | -0.402 |
| Llama-3-8B | **-0.301** | -0.337 | -0.403 |
| Mistral-7B | **-0.217** | -0.394 | -0.437 |
| **Mean** | **-0.289** | **-0.345** | **-0.401** |
| **Std Dev** | 0.047 | 0.045 | 0.031 |

**Interpretation**: All pairwise cosine similarities are negative, indicating the direction vectors point in nearly opposite directions. This strongly supports the tripartite hypothesis.

### 3.2 SAE Clustering (Experiment A)

| Model | Optimal k | Theoretical k | Silhouette | Cluster Purity |
|-------|-----------|---------------|------------|----------------|
| Qwen2.5-7B | 2 | 3 | 0.411 | 0.231 |
| Mistral-7B | 2 | 3 | 0.259 | 0.250 |
| Llama-3-8B | 2 | 3 | 0.224 | 0.250 |
| Llama-3.1-8B | 2 | 3 | 0.211 | 0.250 |
| **Mean** | **2.0** | **3** | **0.276** | **0.245** |

**Interpretation**: SAE clustering consistently finds k=2 clusters instead of the theoretical k=3. This suggests either:
1. Two empathy types are more similar than theorized
2. Dataset size (240 samples) is insufficient for fine-grained clustering
3. The tripartite structure exists but requires more specialized probing

### 3.3 Convergence Assessment

| Metric | Value | Interpretation |
|--------|-------|----------------|
| K-match rate | 0% | SAEs find 2 clusters, not 3 |
| Mean silhouette | 0.276 | Moderate cluster quality |
| Mean purity | 0.245 | Low (random = 0.167) |
| **Conclusion** | **WEAK_CONVERGENCE** | Methods partially agree |

---

## 4. Validation Experiments

To strengthen the findings and test their specificity, we conducted three additional validation experiments on models supported by TransformerLens 1.17.0.

### 4.1 Control Analysis: Empathy vs Non-Empathy Structure

**Question**: Are the negative cosine similarities specific to empathy, or do they appear in any set of distinct response types?

**Method**:
- Extracted activations for both empathy triplets (Cognitive/Affective/Instrumental) and non-empathy controls (3 arbitrary response types per scenario)
- Trained probes and computed cosine similarities for both conditions
- Compared the structure

**Results**:

| Model | Empathy Mean cos | Control Mean cos | Difference |
|-------|------------------|------------------|------------|
| mistral-7b | **-0.484** | -0.490 | +0.006 |
| llama-3-8b | **-0.488** | -0.493 | +0.005 |
| gemma-7b | **-0.494** | -0.496 | +0.002 |
| **Mean** | **-0.489** | **-0.493** | **+0.004** |

**Critical Finding**: The non-empathy control condition shows nearly identical cosine structure to the empathy condition. This suggests:
1. The negative correlations reflect **general response diversity**, not empathy-specific structure
2. Any set of distinct response types will show similar separation
3. The tripartite "separation" may be an artifact of response distinctiveness rather than empathy geometry

### 4.2 Multi-Layer Sweep

**Question**: At which layer does the Cognitive-Affective separation emerge?

**Method**: Extracted activations at layers 4, 8, 12, 16, 20, and 24 for each model.

**Results**:

| Layer | mistral-7b | llama-3-8b | gemma-7b | Mean |
|-------|------------|------------|----------|------|
| 4 | -0.375 | -0.421 | -0.545 | -0.447 |
| 8 | -0.533 | -0.554 | -0.571 | -0.553 |
| 12 | -0.628 | -0.619 | -0.617 | -0.621 |
| 16 | -0.735 | -0.701 | -0.646 | -0.694 |
| 20 | **-0.775** | -0.678 | -0.728 | -0.727 |
| 24 | -0.774 | — | **-0.769** | -0.771 |

**Finding**: Separation increases monotonically with layer depth, peaking around layers 20-24. This is consistent with higher layers encoding more abstract, task-relevant features.

### 4.3 AUROC Analysis

**Question**: Can linear probes perfectly separate empathy subtypes?

**Method**: 5-fold cross-validated AUROC for binary classification between each pair of empathy types.

**Results**:

| Model | Cog vs Aff | Cog vs Instr | Aff vs Instr | Mean |
|-------|------------|--------------|--------------|------|
| mistral-7b | 1.000 | 1.000 | 1.000 | **1.000** |
| llama-3-8b | 1.000 | 1.000 | 1.000 | **1.000** |
| gemma-7b | 1.000 | 1.000 | 1.000 | **1.000** |

**Finding**: Perfect linear separability across all models and pairs. This confirms empathy subtypes are linearly decodable, though the control analysis suggests this may reflect response distinctiveness rather than empathy-specific encoding.

### 4.4 Implications for Original Findings

The validation experiments introduce important caveats:

| Original Claim | Validation Finding | Revised Interpretation |
|----------------|-------------------|------------------------|
| Empathy subtypes occupy distinct subspaces | Control responses show same structure | May reflect response diversity, not empathy-specific geometry |
| cos(Cog,Aff) < 0.5 confirms tripartite hypothesis | Control cosines equally negative | Separation not specific to empathy |
| Architecture-independent pattern | Confirmed | Still valid |
| Linear separability (AUROC=1.0) | Confirmed | Still valid |

**Revised Conclusion**: While empathy subtypes are clearly distinguishable by linear probes, the control analysis reveals this separability is not unique to empathy. The negative cosine similarities appear to be a general property of any set of meaningfully distinct response types, rather than evidence for empathy-specific neural circuitry.

---

## 5. Causal Validation Experiments

Following the control analysis, we conducted two additional experiments to test whether the empathy directions have any causal or statistical significance beyond what random groupings would produce.

### 5.1 Null Distribution Test

**Question**: Are empathy cosines statistically different from random label assignments?

**Method**:
- Computed empathy cosines (Cognitive/Affective/Instrumental)
- Computed control cosines (non-empathy response types)
- Generated null distribution by randomly permuting labels 20 times
- Compared empathy/control means to null distribution via Z-score

**Results** (pythia-1.4b, layer 12):

| Condition | Mean Cosine | Z-score vs Null |
|-----------|-------------|-----------------|
| Empathy | -0.480 | **+13.9** |
| Control | -0.486 | **+7.3** |
| Null (random) | -0.494 | — |

**Critical Finding**: The empathy Z-score is **positive**, meaning empathy labels produce **LESS** separation than random label shuffling. This definitively shows:

1. The "separation" we measured is not meaningful
2. Random groupings of responses show stronger negative cosines
3. The empathy/control similarity (~0.006 difference) confirms non-specificity

**Interpretation**: Training logistic regression probes on any grouped data produces negative cosines between direction vectors. The empathy labels don't capture any structure beyond (or even as much as) random assignments. This is a critical negative result.

### 5.2 Steering Test

**Question**: Does steering along the Cognitive-Affective direction vector change response style?

**Method**:
- Extracted Cognitive-Affective direction: mean(Cognitive activations) - mean(Affective activations)
- Applied steering at inference time with α ∈ {-3.0, -1.5, 0.0, +1.5, +3.0}
- Positive α should push toward "Cognitive" (perspective-taking)
- Negative α should push toward "Affective" (emotional warmth)
- Generated responses to 5 held-out scenarios

**Results** (pythia-1.4b, layer 12):

| Alpha | Expected Effect | Observed Effect |
|-------|-----------------|-----------------|
| -3.0 | More affective/emotional | Mixed: some first-person, empathetic phrasing |
| -1.5 | Slightly affective | Inconsistent |
| 0.0 | Baseline | Standard responses |
| +1.5 | Slightly cognitive | Some analytical framing |
| +3.0 | More cognitive/analytical | Mixed: some third-person, but often off-topic |

**Example** (Adoption scenario):

```
α=-3.0: "Thank you so much for sharing your story with us. As a graduate
        student, I am so glad to hear that you are struggling..."

α=0.0:  "We would like to thank this student for sharing their story..."

α=+3.0: "This is a great example of how a DNA test can be used as a tool
        to help people learn about their past..."
```

**Finding**: Steering produces observable changes in output, but:
1. Changes don't clearly map to cognitive vs affective empathy
2. Higher magnitudes often produce incoherent or off-topic responses
3. The "direction" doesn't appear to encode empathy-specific information

### 5.3 Implications of Causal Tests

| Test | Result | Implication |
|------|--------|-------------|
| Null Distribution | Z=+13.9 (wrong direction) | Empathy labels capture LESS structure than random |
| Steering | Inconsistent shifts | Direction doesn't encode empathy-specific information |

**Conclusion**: Both causal validation experiments confirm the control analysis finding. The tripartite "separation" is a methodological artifact of training binary classifiers on grouped text data, not evidence of empathy-specific neural encoding.

---

## 6. Cross-Model Analysis

### 6.1 Consistency of Separation

```
Cognitive-Affective Cosine Similarity by Model:

Qwen2.5-7B    ████████████████████████████████ -0.322
Llama-3.1-8B  ███████████████████████████████░ -0.317
Llama-3-8B    ██████████████████████████████░░ -0.301
Mistral-7B    █████████████████████░░░░░░░░░░░ -0.217
              ─────────────────────────────────
              -0.4        -0.2         0.0
```

**Finding**: Separation is consistent across all architectures, with Qwen showing the strongest and Mistral the weakest Cognitive-Affective distinction.

### 6.2 Ranking of Separations

Across all models, the empathy subtype pairs rank consistently:

1. **Affective-Instrumental** (mean: -0.401) - Most distinct
2. **Cognitive-Instrumental** (mean: -0.345) - Moderately distinct
3. **Cognitive-Affective** (mean: -0.289) - Least distinct (but still strong)

This suggests Instrumental (problem-solving) empathy is the most representationally distinct, while Cognitive and Affective share more features.

### 6.3 Model Family Effects

| Family | Mean cos(Cog,Aff) | Mean Silhouette |
|--------|-------------------|-----------------|
| Llama | -0.309 | 0.218 |
| Qwen | -0.322 | 0.411 |
| Mistral | -0.217 | 0.259 |

Qwen shows both the strongest probe separation and best SAE clustering, suggesting its architecture may be better suited for representing empathy subtypes distinctly.

---

## 7. Discussion

### 7.1 Revised Assessment of Tripartite Hypothesis

The validation experiments significantly revise our interpretation:

**Original Claim**: Empathy subtypes occupy distinct subspaces (H1: Confirmed)
**Revised Assessment**: **H1: Confirmed but non-specific**

While empathy subtypes are indeed separable (cos(Cog, Aff) = -0.29), the control analysis reveals this separation is **not unique to empathy**. Non-empathy response types show identical cosine structure (mean -0.49 for both conditions). This suggests:

1. The negative cosines reflect **response distinctiveness**, not empathy-specific geometry
2. Any sufficiently different response types would show similar separation
3. The finding is real but less meaningful than initially claimed

### 7.2 What the Results Actually Show

| What We Found | What It Means |
|---------------|---------------|
| Negative cosines between empathy subtypes | Responses are distinguishable |
| Same structure in non-empathy controls | Not empathy-specific |
| Perfect AUROC (1.0) | Linear probes work |
| Layer-depth correlation | Higher layers encode more abstract features |

### 7.3 SAE-Probe Convergence (H2: Partially Supported)

The failure of SAEs to recover k=3 clusters remains unexplained but is now less concerning given the control analysis suggests we may not have been measuring empathy-specific structure at all.

### 7.4 Architecture Independence (H3: Confirmed)

The consistency across architectures remains valid—all models show the same separation pattern. However, this may simply confirm that all models learn to distinguish different response types, not that they encode empathy specifically.

### 7.5 Revised Implications

1. **For Safety**: Empathy steering via direction vectors may work but is not targeting "empathy" specifically—just response style
2. **For Alignment**: Models distinguish response types generally, not empathy subtypes specifically
3. **For Methodology**: Control conditions are essential for validating specificity of geometric findings

---

## 8. Conclusions

### 8.1 Summary of Findings

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1: Separation | **Artifact** | cos(Cog,Aff) = -0.29, but null distribution shows random is MORE separated |
| H2: Convergence | **Partial** | SAE k=2 vs theory k=3 |
| H3: Consistency | **Confirmed** | All models show negative cosines (but meaningless) |
| H4: Specificity | **NOT Confirmed** | Control cosines match empathy cosines |
| **H5: Causality** | **NOT Confirmed** | Steering doesn't produce empathy-specific shifts |
| **H6: Statistical** | **REFUTED** | Null distribution Z=+13.9 (wrong direction) |

### 8.2 Main Contributions

1. **Critical negative result**: Definitively showed that probe cosine similarity is NOT a valid measure of concept-specific structure
2. **Null distribution methodology**: Demonstrated how to properly validate geometric claims via random permutation testing
3. **Control condition importance**: Showed that any distinct response types produce identical "separation"
4. **Steering validation**: Confirmed that geometric separation doesn't imply causal influence
5. **Methodological warning**: Representation engineering claims require null distribution testing, not just cosine thresholds

### 8.3 Key Takeaway

**The tripartite separation is a methodological artifact, not evidence of empathy-specific encoding.** The null distribution test definitively shows that random label permutations produce MORE negative cosines than empathy labels (Z=+13.9). This means:

1. Training logistic regression probes on grouped data inherently produces negative cosines
2. The "separation" we measured is less than random chance would produce
3. Steering along these directions doesn't produce empathy-specific behavioral shifts

This is a critical negative result for representation engineering methodology: **cosine similarity between probe directions is not a valid measure of concept-specific structure.**

### 8.4 Limitations

1. **Dataset size**: 240 samples limits SAE training quality
2. **Single layer**: Only analyzed one layer per model (validation added multi-layer)
3. **English only**: Results may not transfer to other languages
4. **Model scale**: Only tested 7-8B models
5. **Control scope**: Only tested one type of non-empathy control

---

## 9. Future Work

Given the control analysis findings, future work should focus on:

1. **Finding Empathy-Specific Signatures**: Design experiments that isolate empathy from general response diversity
2. **Causal Validation**: Ablation experiments to test if directions causally affect empathy quality
3. **More Control Conditions**: Test multiple types of non-empathy controls to bound the specificity
4. **Behavioral Correlation**: Correlate geometric measures with human-rated empathy to validate relevance
5. **Steering Experiments**: Test whether steering along "empathy" directions actually increases empathy vs. just changing response style
6. **Cross-concept Comparison**: Compare empathy geometry to other concepts (humor, formality, etc.) to establish baselines

---

## 10. Appendix

### A. Statistical Summary

```
Probe Cosine Similarities (N=4 models)
─────────────────────────────────────────
                    Mean     Std     Min      Max
Cog-Aff           -0.289   0.047  -0.322   -0.217
Cog-Instr         -0.345   0.045  -0.394   -0.287
Aff-Instr         -0.401   0.031  -0.437   -0.361

SAE Clustering (N=4 models)
─────────────────────────────────────────
                    Mean     Std     Min      Max
Optimal K           2.00    0.00    2.00     2.00
Silhouette          0.28    0.09    0.21     0.41
Cluster Purity      0.25    0.01    0.23     0.25
```

### B. Effect Size

Using Cohen's d for the Cognitive-Affective separation:
- d = |mean| / std = 0.289 / 0.047 = **6.15** (very large effect)

### C. Data Availability

All data and code available at:
https://github.com/marcosantar93/empathetic-language-bandwidth

```
experiments/tripartite/
├── data/                           # Input datasets
├── results/                        # Per-model results
│   ├── qwen2.5-7b/
│   ├── mistral-7b/
│   ├── llama-3-8b/
│   ├── llama-3.1-8b/
│   ├── validation_*.json           # Validation experiment results
│   ├── null_distribution_pythia.json  # Null distribution test
│   └── steering_test_pythia.json   # Steering test results
└── scripts/                        # Analysis code
    ├── run_all_validation.py       # Control/multilayer/AUROC
    ├── run_null_distribution.py    # Null distribution test
    └── run_steering_test.py        # Steering test
```

---

## References

1. Burns, C., et al. (2023). "Discovering Latent Knowledge in Language Models." *ICLR*.
2. Zou, A., et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency." *ArXiv*.
3. Li, K., et al. (2024). "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model." *NeurIPS*.
4. Templeton, A., et al. (2024). "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." *Anthropic*.
5. Davis, M. H. (1983). "Measuring individual differences in empathy: Evidence for a multidimensional approach." *Journal of Personality and Social Psychology*.

---

*Report generated: January 29, 2026*
*Author: Marco Santarcangelo*
*Contact: https://marcosantar.com*
