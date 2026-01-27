# Empathetic Language Bandwidth - Project Summary

## Overview

This project investigates how LLMs represent and generate empathetic language using mechanistic interpretability techniques (activation geometry, probes, SAEs). The research is organized in two phases.

---

## Phase 1: Bandwidth Measurement (COMPLETE)

**Status:** Finished January 18, 2026

### Goal
Measure the "empathetic bandwidth" of LLMs - how many dimensions in activation space encode empathy, and how steerable those dimensions are.

### Models Tested
| Model | Bandwidth Score | Dimensions | Steering Range | AUROC |
|-------|-----------------|------------|----------------|-------|
| Gemma2-9B | **136.6** | 16 | 8.5 | 0.950 |
| Llama-3.1-8B | 127.0 | 14 | 9.1 | 0.874 |
| DeepSeek-R1-7B | 92.0 | 11 | 8.4 | 0.856 |
| Qwen2.5-7B | 67.3 | 10 | 6.7 | 0.835 |
| Mistral-7B | 36.3 | 6 | 6.0 | 0.829 |

### Key Findings
- **109% variation** in empathetic bandwidth across models
- **2.8x larger** than syntactic complexity control baseline
- **80% SAE-PCA agreement** validates the measurement approach
- **87% transfer success** across contexts (crisis support → technical assistance)
- **Cohen's d = 2.41** (large effect size)

### Methodology
1. Train logistic regression probes on layer 24 activations
2. Measure dimensionality via PCA effective rank (90% variance threshold)
3. Compute steering range (max α before coherence drops below 0.7)
4. Cross-validate with SAEs
5. Test transfer across contexts

### Results Location
- `results/empathy/empathy_geometry_report_*.md` - Full technical reports
- `results/empathy/all_results_*.json` - Complete numerical data

---

## Phase 2: Tripartite Decomposition (IN PROGRESS)

**Status:** Partially complete (2/4 models)

### Research Question
Does empathy decompose into distinct subspaces?
- **Cognitive empathy:** Perspective-taking, understanding mental states
- **Affective empathy:** Emotional resonance, warmth
- **Instrumental:** Problem-solving (control)

### Methodology
**Dual-path validation:**
- **Experiment A:** Let SAE discover natural clusters (geometry-driven)
- **Experiment B:** Train probes for Cog/Aff/Instrumental (theory-driven)
- **Convergence test:** Do paths agree?

### Datasets Generated
| File | Description | Size |
|------|-------------|------|
| `triplets_raw.json` | 90 scenarios × 3 responses | 223 KB |
| `triplets_filtered.json` | Hand-filtered best quality | 241 KB |
| `controls_non_empathy.json` | 60 non-empathy emotional | 83 KB |
| `controls_valence_stripped.json` | 90 valence-stripped | 135 KB |

### Models & Status
| Model | Status | Notes |
|-------|--------|-------|
| Qwen2.5-7B | Complete (6-7 min) | Results on terminated pod |
| Mistral-7B | Complete (8 min) | Partial results downloaded |
| Llama-3-8B | Failed | Bug in activation extraction |
| Llama-3.1-8B | Not started | SSH never became available |

### Preliminary Results (Mistral-7B)
```
Convergence: WEAK_CONVERGENCE
Cluster purity: 0.211
Silhouette score: 0.219

Probe cosine similarities:
- cos(Cognitive, Affective): -0.219
- cos(Cognitive, Instrumental): -0.392
- cos(Affective, Instrumental): -0.439
```

The negative cosine similarities indicate separation between empathy subtypes.

### Known Issues
1. **Results lost:** Pods terminated before downloading Qwen results
2. **Llama-3-8B:** Extraction reports success but doesn't create files
3. **Llama-3.1-8B:** RunPod API reported 0s uptime despite 1+ hour runtime

---

## Infrastructure

### Local (M1 Mac)
- Dataset generation via Anthropic API
- Manual filtering
- Analysis and reporting

### Remote (RunPod GPU)
- RTX A5000 @ $0.16/hr
- Docker: `pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- SSH automation via `runpod_empathy.py`

### Pipeline Scripts
```
experiments/tripartite/scripts/
├── generate_triplets.py      # LOCAL - dataset generation
├── generate_controls.py      # LOCAL - control conditions
├── run_all_gpu.py            # REMOTE - orchestrator
├── extract_activations.py    # REMOTE - TransformerLens
├── train_saes.py             # REMOTE - Experiment A
├── train_probes.py           # REMOTE - Experiment B
└── convergence_analysis.py   # REMOTE - validation
```

---

## Cost Summary

| Phase | Infrastructure | Cost |
|-------|---------------|------|
| Phase 1 | RunPod + EC2 | ~$2-3 |
| Phase 2 (partial) | RunPod (4 pods × 1hr) | ~$0.65 |
| Data generation | Anthropic API | ~$2 |
| **Total to date** | | **~$5-6** |

Execution was 10-12x faster than estimated due to optimized GPU code.

---

## Success Criteria Progress

| Criterion | Target | Status |
|-----------|--------|--------|
| Separation | cos(Cog, Aff) < 0.5 | Achieved (Mistral: -0.22) |
| Convergence | SAE ↔ Probe > 0.5 | Weak (0.22) |
| Specificity | Controls differ | Untested |
| Causality | Ablation degrades quality | Not implemented |

---

## Next Steps

1. **Re-run Phase 2** for Qwen and download results immediately
2. **Debug** Llama-3-8B activation extraction
3. **Retry** Llama-3.1-8B with Jupyter approach
4. **Run control analysis** to validate specificity
5. **Implement ablation** experiments for causality
6. **Generate final report** with all 4 models

---

## Key Files

- `CLAUDE.md` - Project instructions
- `README.md` - Public documentation
- `experiments/tripartite/FINAL_SESSION_SUMMARY.md` - Detailed session log
- `results/empathy/empathy_geometry_report_*.md` - Phase 1 report
- `docs/FUTURE_EXPERIMENTS.md` - 10 follow-up experiment ideas

---

*Last updated: January 27, 2026*
