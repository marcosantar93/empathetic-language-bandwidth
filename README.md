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

## Repository Structure

```
empathetic-language-bandwidth/
├── src/
│   ├── empathy_experiment_main.py          # Main experiment pipeline
│   ├── generate_synthetic_empathy_results.py  # Synthetic data generator (for testing)
│   ├── analyze_empathy_results.py          # Analysis and statistics
│   └── create_empathy_report.py            # Report generation
├── data/
│   └── prompts/                            # Empathetic/neutral prompt pairs (TBD)
├── results/
│   └── empathy/                            # Experiment results and reports
├── docs/
│   ├── empathy_geometry_proposal.md        # Original research proposal
│   └── FUTURE_EXPERIMENTS.md               # 10 follow-up experiment ideas
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

See [FUTURE_EXPERIMENTS.md](docs/FUTURE_EXPERIMENTS.md) for 10 prioritized follow-up experiments:

1. **Human Evaluation** - Validate bandwidth correlates with helpfulness
2. **Layer-wise Profiling** - Track empathy emergence across layers
3. **Causal Intervention** - Test via activation patching
4. **Scaling to 70B** - Test larger models
5. **Extended Transfer** - 10 diverse contexts
6. **Toxicity Trade-off** - Empathy vs. safety
7. **Fine-tuning** - Can we increase bandwidth?
8. **Multimodal** - Vision + language empathy
9. **Adversarial** - Over-steering failure modes
10. **Cross-lingual** - Transfer across languages

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
