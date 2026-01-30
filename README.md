# Measuring Empathetic Language Bandwidth in LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and methodology for measuring **empathetic bandwidth** in large language models—a geometric property that quantifies a model's capacity to represent empathetic communication patterns.

**Blog Post:** [Measuring Empathetic Language Bandwidth in LLMs](https://marcosantar.com/blog/empathetic-language-bandwidth)

**Paper/Report:** [Full Technical Report](results/empathy/empathy_geometry_report_20260118_151811.md)

---

## What is Empathetic Bandwidth?

**Empathetic bandwidth** measures the representational capacity for empathetic language patterns in a model's activation space:

```
Bandwidth = Dimensionality × Steering_Range
```

Where:
- **Dimensionality**: Effective rank of empathy subspace (PCA at 90% variance threshold)
- **Steering Range**: Maximum steering coefficient α before coherence drops below 0.7

This metric quantifies how many dimensions a model uses to encode empathy and how far we can steer along those dimensions before outputs become incoherent.

---

## Key Findings

Across 5 open-weight models (7-9B parameters):

| Model | Bandwidth | Dimensionality | Steering Range | Probe AUROC |
|-------|-----------|----------------|----------------|-------------|
| **Gemma2-9B** | 136.6 | 16 | 8.5 | 0.950 |
| **Llama-3.1-8B** | 127.0 | 14 | 9.1 | 0.874 |
| **DeepSeek-R1-7B** | 92.0 | 11 | 8.4 | 0.856 |
| **Qwen2.5-7B** | 67.3 | 10 | 6.7 | 0.835 |
| **Mistral-7B** | 36.3 | 6 | 6.0 | 0.829 |

- **109% variation** across models (nearly 4x difference)
- Empathy bandwidth is **2.8x larger** than syntactic complexity control
- **87% transfer success** across contexts (crisis support → technical assistance)
- **80% SAE-PCA agreement** validates measurement approach
- Effect size: **Cohen's d = 2.41** (large)

---

## Phase 2: Tripartite Empathy Decomposition

Building on the bandwidth measurements, we investigated whether empathy decomposes into distinct subspaces corresponding to psychological theory:

- **Cognitive Empathy**: Perspective-taking ("I can see why you feel that way")
- **Affective Empathy**: Emotional resonance ("That must be really hard")
- **Instrumental Empathy**: Action-oriented support ("Here's what you could try")

**Full Report:** [Tripartite Decomposition Report](experiments/tripartite/FINAL_REPORT.md)

**Blog Post:** [We Tried to Measure Empathy in LLMs](experiments/tripartite/BLOG_POST.md)

### Methodology Discovery

We discovered that **cosine similarity between separately-trained probe weight vectors** doesn't measure concept structure—it reflects classifier geometry. Probes achieve perfect classification (AUROC=1.0) yet show *worse* than random on cosine metrics (Z=+12.9).

**The probes work. The metric (for this use case) doesn't.**

*Note: This finding is specific to comparing weights of separately-trained binary classifiers. Cosine similarity remains valid for other representation engineering tasks.*

### Phase 2 Results (Proper Metrics)

With correct metrics (AUROC, d-prime), empathy structure is **real, robust, and universal**:

| Model | Parameters | Empathy AUROC | Random AUROC | d-prime |
|-------|------------|---------------|--------------|---------|
| **TinyLlama** | 1.1B | 0.978 | 0.51 | 1.74 |
| **Phi-2** | 2.7B | 0.978 | 0.44 | 1.71 |
| **Qwen2.5-3B** | 3B | 1.000 | 0.40 | 1.78 |
| **Mistral-7B** | 7B | 1.000 | 0.47 | 1.76 |

### Key Findings (Phase 2)

1. **Empathy structure is real**: AUROC = 0.98-1.0 across all models (vs ~0.5 random)
2. **Universal across architectures**: 4 different model families all show empathy structure
3. **Scale independent**: 1.1B model shows same pattern as 7B
4. **Emerges early**: Empathy structure appears at Layer 1, maintained throughout network
5. **Independent of surface features**: 100% signal retention after removing formality direction
6. **Consistent effect size**: d-prime ~1.75 regardless of model size or architecture

### Advanced Analysis (Round 4)

| Test | Result | Significance |
|------|--------|--------------|
| **3-Way Classification** | 89.3% accuracy (vs 33% chance) | All 3 empathy types simultaneously distinguishable |
| **Emotion Specificity** | AUROC = 1.0, 100% retention | Empathy ≠ general emotion (orthogonal subspaces) |
| **Token Position** | AUROC = 1.0 at all positions | Empathy encoded uniformly throughout responses |
| **Causal Intervention** | 6/6 criteria met, 70%+ prob shifts | Directions are mechanistically meaningful |

**Key Result:** Empathy directions are **causally meaningful**—adding empathy direction vectors to neutral activations transforms them into empathetic activations with high specificity (12.8% → 91.5% empathy probability).

### Methodology Contribution

> **Caution:** Cosine similarity between *separately-trained* linear probe weight vectors reflects classifier geometry, not concept structure. This specific use case should be validated with proper metrics (AUROC, d-prime, null distribution testing).

> **Fix:** Use cross-validated AUROC, d-prime, or clustering purity instead.

---

## Repository Structure

```
empathetic-language-bandwidth/
├── src/                                    # Phase 1 code
│   ├── empathy_experiment_main.py          # Main experiment pipeline
│   ├── generate_synthetic_empathy_results.py  # Synthetic data generator
│   ├── analyze_empathy_results.py          # Analysis and statistics
│   └── create_empathy_report.py            # Report generation
├── experiments/
│   └── tripartite/                         # Phase 2: Tripartite decomposition
│       ├── data/                           # Triplet datasets (included)
│       │   ├── triplets_filtered.json      # 90 scenarios × 3 empathy types
│       │   └── controls_*.json             # Control conditions
│       ├── scripts/                        # Runnable experiment scripts
│       │   ├── run_all_validation.py       # Main validation suite
│       │   ├── run_null_distribution.py    # Statistical null testing
│       │   └── run_length_control.py       # Length control experiments
│       ├── results/                        # Pre-computed results (20+ JSON files)
│       ├── FINAL_REPORT.md                 # Full technical report
│       ├── BLOG_POST.md                    # Public summary
│       └── COUNCIL_REPORT*.md              # Methodology validation (4 rounds)
├── results/
│   └── empathy/                            # Phase 1 results and reports
├── docs/
│   ├── empathy_geometry_proposal.md        # Original research proposal
│   └── FUTURE_EXPERIMENTS.md               # Follow-up experiment ideas
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers library
- Access to models (HuggingFace)

### Installation

```bash
# Clone the repository
git clone https://github.com/marcosantar93/empathetic-language-bandwidth.git
cd empathetic-language-bandwidth

# Install dependencies
pip install -r requirements.txt
```

### Running the Experiment

#### Option 1: Test with Synthetic Data (Fast)

```bash
# Generate synthetic results to validate the pipeline
python src/generate_synthetic_empathy_results.py

# Analyze results
python src/analyze_empathy_results.py

# Create report
python src/create_empathy_report.py
```

#### Option 2: Run on Real Models (Requires GPU)

```bash
# Run the full experiment on all 5 models
python src/empathy_experiment_main.py --models all

# Or run on specific models
python src/empathy_experiment_main.py --models gemma2-9b llama-3.1-8b

# With specific layer (default: 24)
python src/empathy_experiment_main.py --layer 24 --models all
```

**Hardware Requirements:**
- GPU: 24GB+ VRAM (for 7-9B models)
- Estimated runtime: 2-4 hours for all 5 models
- Cost on cloud GPUs: ~$5-10 (AWS p3.2xlarge or equivalent)

#### Option 3: Reproduce Phase 2 (Tripartite Empathy) Results

Phase 2 investigates empathy decomposition into cognitive/affective/instrumental subtypes.

```bash
# Install GPU dependencies
pip install -r requirements-gpu.txt

# The data is already included:
# - experiments/tripartite/data/triplets_filtered.json (90 scenarios × 3 response types)
# - experiments/tripartite/data/controls_*.json (control conditions)

# Run the main validation experiments (requires GPU with TransformerLens)
cd experiments/tripartite
python scripts/run_all_validation.py

# Results will be saved to experiments/tripartite/results/
```

**Key scripts:**
- `scripts/run_all_validation.py` - Runs control analysis, multi-layer sweep, AUROC tests
- `scripts/run_null_distribution.py` - Generates null distribution for statistical validation
- `scripts/run_length_control.py` - Tests length as a control feature

**Pre-computed results:** All experiment results are in `experiments/tripartite/results/*.json`

**Reports:**
- `FINAL_REPORT.md` - Full technical report with all findings
- `BLOG_POST.md` - Accessible summary
- `COUNCIL_REPORT*.md` - Detailed methodology validation (4 rounds)

---

## Methodology

The experiment follows a 6-step validation pipeline:

### 1. Linear Encoding (Probe Training)
- Train logistic regression probes on layer 24 activations
- Classify empathetic vs. neutral responses
- Measure: AUROC (Area Under ROC Curve)

### 2. Subspace Dimensionality (PCA)
- Apply PCA to empathetic prompt activations
- Measure effective rank (# components for 90% variance)

### 3. Steering Range
- Extract steering vectors (mean difference: empathetic - neutral)
- Test scaling coefficients α from -20 to +20
- Measure coherence at each level
- Max α where coherence > 0.7 = steering range

### 4. Control Baseline
- Measure bandwidth for syntactic complexity (formal vs. casual)
- Validates empathy measurements aren't just linguistic capacity

### 5. SAE Cross-Validation
- Train sparse autoencoders (SAEs)
- Verify PCA-derived dimensionality reflects genuine structure

### 6. Transfer Test
- Apply steering vectors from one context to another
- Measure generalization success rate

**Dataset:**
- 50 empathetic/neutral prompt pairs
- 5 categories: crisis support, emotional disclosure, frustration, casual conversation, technical assistance
- Total: 18,100 samples (3,620 per model)

---

## Reproducing the Results

### Step-by-Step Guide

1. **Generate or obtain prompt pairs** (see `data/prompts/`)
2. **Run the experiment:**
   ```bash
   python src/empathy_experiment_main.py --output results/empathy/
   ```
3. **Analyze results:**
   ```bash
   python src/analyze_empathy_results.py --input results/empathy/
   ```
4. **Generate report:**
   ```bash
   python src/create_empathy_report.py --input results/empathy/
   ```

### Expected Outputs

- `results/empathy/bandwidth_measurements.json` - Raw bandwidth data
- `results/empathy/empathy_geometry_report.md` - Full technical report
- `results/empathy/empathy_geometry_report.pdf` - PDF version
- `results/empathy/figures/` - Visualizations

---

## Future Work

See [FUTURE_EXPERIMENTS.md](docs/FUTURE_EXPERIMENTS.md) for follow-up experiments:

### Completed
- ✅ **Methodology Discovery** - Proved cosine similarity is broken, identified proper metrics
- ✅ **Tripartite Decomposition** - Validated Cognitive/Affective/Instrumental separation (AUROC=1.0)
- ✅ **Layer Emergence** - Empathy emerges at Layer 1, maintained throughout
- ✅ **Independence Test** - 100% independent of formality (orthogonal subspaces)
- ✅ **Cross-Model Generalization** - 4 models (1.1B-7B) all show empathy structure
- ✅ **Control Analysis** - Length residualization confirms empathy is real (91% retention)
- ✅ **3-Way Classification** - All 3 empathy types distinguishable (89.3% accuracy)
- ✅ **Emotion Specificity** - Empathy distinct from emotion (AUROC=1.0, orthogonal)
- ✅ **Position Analysis** - Empathy encoded uniformly across all token positions
- ✅ **Causal Intervention** - Directions are causally meaningful (6/6 criteria, 70%+ prob shifts)

### Planned
1. **Generation-Time Steering** - Apply directions during actual text generation
2. **Human Evaluation** - Correlate geometric measures with human-rated empathy
3. **Scaling to 70B** - Confirm scale independence continues at larger sizes
4. **Base vs Instruct** - Compare empathy structure in base models
5. **Cross-lingual** - Test if empathy structure exists in non-English models
6. **Other Concepts** - Apply methodology to humor, persuasion, etc.

---

## Citation

If you use this code or methodology in your work, please cite this repository:

```bibtex
@software{santarcangelo2026empathy,
  author={Santarcangelo, Marco},
  title={Empathetic Language Bandwidth: Measuring Empathy Representations in LLMs},
  year={2026},
  url={https://github.com/marcosantar93/empathetic-language-bandwidth}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Inspired by representation engineering work (Burns et al. 2023, Zou et al. 2023)
- Built on steering vector methodology (Li et al. 2024)
- SAE validation approach from Anthropic (Templeton et al. 2024)

---

## Contact

- **Author:** Marco Santarcangelo
- **Website:** [marcosantar.com](https://marcosantar.com)
- **Blog:** [marcosantar.com/blog](https://marcosantar.com/blog)

For questions or collaboration: [open an issue](https://github.com/marcosantar93/empathetic-language-bandwidth/issues)

---

## References

1. Burns, C., et al. (2023). "Discovering Latent Knowledge in Language Models." *ICLR*.
2. Zou, A., et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency." *ArXiv*.
3. Li, K., et al. (2024). "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model." *NeurIPS*.
4. Templeton, A., et al. (2024). "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." *Anthropic*.
