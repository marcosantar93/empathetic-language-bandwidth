# Empathetic Language Bandwidth Research

## Project Overview

Measuring empathetic bandwidth in LLMs via activation geometry. Testing whether empathy decomposes into distinct subspaces (Cognitive vs Affective) and whether this structure is model-dependent.

**Phase 1:** Bandwidth measurement — DONE (synthetic validation)
**Phase 2:** Tripartite decomposition — IN PROGRESS

---

## Execution Environment

### Local (M1 Mac)
- Dataset generation (Anthropic API)
- Human filtering of triplets
- Analysis, statistics, reporting
- No GPU, no torch

### Remote (EC2 via docker_ec2_builder)
- Activation extraction (TransformerLens)
- SAE training (SAE Lens)
- Steering and ablation experiments
- Requires: 24GB+ VRAM, CUDA
- Builder tool: `~/docker_ec2_builder`

### Handoff Protocol
1. Local: Generate datasets → `experiments/tripartite/data/`
2. Push to GitHub
3. EC2: Pull repo, run `Dockerfile.gpu`, execute experiments
4. EC2: Push results to `experiments/tripartite/results/`
5. Local: Pull results, run analysis and reporting

---

## Project Structure
```
empathetic-language-bandwidth/
├── CLAUDE.md                    # This file
├── README.md                    # Public documentation
├── requirements.txt             # Local dependencies (no GPU)
├── requirements-gpu.txt         # EC2 dependencies
├── Dockerfile.gpu               # Container for remote execution
├── src/                         # Phase 1 code
│   ├── empathy_experiment_main.py
│   ├── analyze_empathy_results.py
│   ├── create_empathy_report.py
│   └── generate_synthetic_empathy_results.py
├── results/empathy/             # Phase 1 results (synthetic)
├── experiments/tripartite/      # Phase 2 (to create)
│   ├── data/
│   │   ├── triplets_raw.json
│   │   ├── triplets_filtered.json
│   │   ├── controls_non_empathy.json
│   │   └── controls_valence_stripped.json
│   ├── scripts/
│   │   ├── generate_triplets.py      # LOCAL
│   │   ├── generate_controls.py      # LOCAL
│   │   ├── extract_activations.py    # REMOTE
│   │   ├── train_probes.py           # REMOTE
│   │   ├── train_saes.py             # REMOTE
│   │   ├── convergence_analysis.py   # LOCAL
│   │   └── run_all_gpu.py            # REMOTE entrypoint
│   ├── activations/             # REMOTE generates
│   ├── saes/                    # REMOTE generates
│   ├── results/
│   │   ├── experiment_a/        # SAE-driven discovery
│   │   └── experiment_b/        # Theory-driven probes
│   └── figures/
└── docs/
```

---

## Reusable Code

From `../crystallized-safety/src/` (copy as needed):
- `extraction.py` — TransformerLens activation hooks
- `steering.py` — direction vectors, ablation methods
- `evaluation.py` — probe training, AUROC calculation

---

## Phase 2: Tripartite Decomposition

### Research Question
Does empathy decompose into Cognitive (perspective-taking) vs Affective (warmth)?

### Methodology
**Dual-path validation:**
- Experiment A: Let SAE discover natural clusters (geometry-driven)
- Experiment B: Train probes for Cog/Aff/Instrumental (theory-driven)
- Convergence test: Do paths agree?

### Dataset
- 90 scenarios × 3 responses (Cognitive/Affective/Instrumental)
- Controls: non-empathy emotional (60) + valence-stripped (90)
- Total: ~420 triplets

### Models
- Gemma2-9B
- Llama-3.1-8B
- Qwen2.5-7B
- Mistral-7B
- DeepSeek-R1-7B

### Key Metrics
- Probe AUROC (linear separability)
- Cosine(Cognitive, Affective) — <0.5 means separation
- SAE cluster count vs theory-predicted (4)
- Convergence: SAE clusters ↔ probe vectors alignment

---

## Immediate Tasks (Local)

1. Create `experiments/tripartite/` directory structure
2. Write `generate_triplets.py` with 90 scenarios + rubric
3. Write `generate_controls.py` for both control conditions
4. Run dataset generation (~$2, ~30 min)
5. Manual filtering: keep 90 best triplets
6. Create `Dockerfile.gpu` and `requirements-gpu.txt`

## Next Tasks (Remote)

7. Build container via docker_ec2_builder
8. Extract activations for all models
9. Train SAEs (Experiment A)
10. Train probes (Experiment B)
11. Run convergence analysis
12. Generate report and figures

---

## Commands Reference

### Local
```bash
source venv/bin/activate
python experiments/tripartite/scripts/generate_triplets.py --output data/triplets_raw.json
python experiments/tripartite/scripts/generate_controls.py non-empathy --output data/controls_non_empathy.json
```

### Remote (via docker_ec2_builder)
```bash
~/docker_ec2_builder/build_and_run.py \
  --dockerfile Dockerfile.gpu \
  --context . \
  --instance-type g5.xlarge \
  --cmd "python experiments/tripartite/scripts/run_all_gpu.py"
```

---

## Success Criteria

1. **Separation:** cos(Cognitive, Affective) < 0.5 for majority of models
2. **Convergence:** SAE clusters align with probes (mean cosine > 0.5)
3. **Specificity:** Control datasets show different structure than empathy
4. **Causality:** Ablating directions degrades empathetic quality

---

## Links

- Blog: https://marcosantar.com/blog/empathetic-language-bandwidth
- GitHub: https://github.com/marcosantar93/empathetic-language-bandwidth
- Related: ../crystallized-safety (safety steering research)
