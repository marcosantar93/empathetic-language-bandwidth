# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research measuring empathetic bandwidth in LLMs via activation geometry. Two phases:

- **Phase 1** (complete): Bandwidth measurement across 5 models (7-9B). Synthetic validation pipeline in `src/`.
- **Phase 2** (complete): Tripartite empathy decomposition (Cognitive/Affective/Instrumental). Experiments in `experiments/tripartite/`. Key finding: cosine similarity between separately-trained probe weight vectors reflects classifier geometry, not concept structure. AUROC and d-prime are the correct metrics.

## Split Execution Environment

**Local (M1 Mac — no GPU, no torch):** dataset generation via Anthropic API, human filtering, analysis, reporting.

**Remote (GPU — EC2 or RunPod):** activation extraction (TransformerLens), SAE training (SAE Lens), steering/ablation. Requires 24GB+ VRAM.

Handoff: local generates data → push → remote runs GPU experiments → push results → local analyzes.

## Commands

```bash
# Activate local env
source venv/bin/activate

# Phase 1: synthetic pipeline (local, no GPU)
python src/generate_synthetic_empathy_results.py
python src/analyze_empathy_results.py
python src/create_empathy_report.py

# Phase 1: real models (GPU required)
python src/empathy_experiment_main.py --models all
python src/empathy_experiment_main.py --models gemma2-9b llama-3.1-8b --layer 24

# Phase 2: dataset generation (local, uses Anthropic API)
python experiments/tripartite/scripts/generate_triplets.py --output ../data/triplets_raw.json --count 90
python experiments/tripartite/scripts/generate_controls.py non-empathy --output ../data/controls_non_empathy.json

# Phase 2: GPU experiments (remote only)
python experiments/tripartite/scripts/run_all_validation.py      # Main validation suite
python experiments/tripartite/scripts/run_null_distribution.py   # Statistical null testing
python experiments/tripartite/scripts/run_length_control.py      # Length confound control
python experiments/tripartite/scripts/run_steering_test.py       # Causal intervention

# Phase 2: RunPod deployment
python3 experiments/tripartite/runpod_empathy.py              # Launch pods for all models
python3 experiments/tripartite/runpod_empathy.py --status     # Check pod status
python3 experiments/tripartite/runpod_empathy.py --cleanup    # Terminate pods

# Docker (EC2 path)
~/docker_ec2_builder/build_and_run.py \
  --dockerfile Dockerfile.gpu --context . \
  --instance-type g5.xlarge \
  --cmd "python experiments/tripartite/scripts/run_all_gpu.py"
```

## Architecture

### Phase 1 (`src/`)
- `empathy_experiment_main.py` — Full experiment pipeline. Uses HuggingFace `transformers` directly. `EmpathyExperiment` class loads model, extracts activations at a layer (default 24), trains probes, measures bandwidth = dimensionality × steering range.
- `generate_synthetic_empathy_results.py` — Creates synthetic results to validate the analysis pipeline without a GPU.
- `analyze_empathy_results.py` / `create_empathy_report.py` — Post-hoc analysis and report generation.

### Phase 2 (`experiments/tripartite/`)
- **Data pipeline (local):** `generate_triplets.py` uses Anthropic API to create 90 scenarios × 3 response types. `generate_controls.py` creates non-empathy emotional and valence-stripped controls. `filter_triplets.py` for human-in-the-loop filtering.
- **GPU experiments (remote):** `run_all_validation.py` is the main validation entry point — self-contained, loads models via TransformerLens `HookedTransformer`, extracts activations with `run_with_cache`, trains logistic regression probes. `extract_activations.py`, `train_probes.py`, `train_saes.py` are the modular GPU scripts. `run_all_gpu.py` is the Docker/EC2 entrypoint.
- **RunPod:** `runpod_empathy.py` launches GPU pods via RunPod API. Needs `RUNPOD_API_KEY` and `HF_TOKEN` env vars.
- **Results:** JSON files in `experiments/tripartite/results/`, organized per-model for probe results and flat for validation experiments.
- **Reports:** `FINAL_REPORT.md`, `BLOG_POST.md`, `COUNCIL_REPORT*.md` (4 rounds of methodology validation).

### Key patterns
- Activation extraction: last-token hidden state at specified layer. Phase 1 uses `output_hidden_states=True`, Phase 2 uses TransformerLens hooks (`blocks.{layer}.hook_resid_post`).
- Probes: sklearn `LogisticRegression` with `cross_val_score` and `roc_auc_score`.
- Model name mapping: short names (e.g. `mistral-7b`) mapped to HuggingFace IDs in dicts at top of scripts.

## Dependencies

- `requirements.txt` — Local (includes torch but used for analysis only on M1)
- `requirements-gpu.txt` — Remote GPU (adds `transformer-lens`, `sae-lens`, `nnsight`, `accelerate`)
- `Dockerfile.gpu` — PyTorch 2.1 + CUDA 12.1 container. Accepts `HF_TOKEN` build arg for gated models.

## Related

- Sibling repo `../crystallized-safety/src/` has reusable code: `extraction.py`, `steering.py`, `evaluation.py`.
