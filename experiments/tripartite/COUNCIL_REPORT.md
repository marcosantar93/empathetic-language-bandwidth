# Research Council Report: Tripartite Empathy Study

**Date:** January 29, 2026
**Cycles Completed:** 3
**Status:** Definitive Finding Achieved

---

## Executive Summary

Through a structured council-driven research process, we discovered that **cosine similarity between probe weight vectors is not a valid measure of concept-specific structure in neural networks**. However, empathy structure IS real and survives rigorous validation using proper metrics.

---

## Council Process

Each cycle followed this protocol:
1. **Proposal** - PI proposes experiment based on current findings
2. **Review** - Statistician, Engineer, and Devil's Advocate provide critiques
3. **Consensus** - Green light only when all concerns addressed
4. **Execution** - Run experiment on GPU (Mistral-7B, RunPod)
5. **Analysis** - Interpret results and plan next cycle

---

## Cycle 1: Alternative Metrics Comparison

### Question
If cosine similarity (between separate probes) doesn't measure concept structure, what metrics DO work?

### Council Deliberation
- **PI**: Proposed testing 4 alternative metrics from the literature
- **Statistician**: Required each metric to beat random permutation baseline
- **Engineer**: Recommended d-prime, probe agreement, clustering purity, AUROC
- **Devil's Advocate**: "What if ALL metrics fail? We need to be ready for that."
- **Consensus**: Green light to test all 4 metrics

### Experiment
Computed 4 metrics for:
- Empathy labels (Cognitive vs Affective vs Instrumental)
- Random permuted labels (100 permutations)

### Results

| Metric | Empathy | Random | Empathy Wins? |
|--------|---------|--------|---------------|
| Mean Diff Projection (d') | **12.1** | 1.0 | YES |
| Probe Agreement | **0.96** | 0.70 | YES |
| Clustering Purity | **0.97** | 0.42 | YES |
| AUROC | **1.00** | 0.44 | YES |

### Finding
**All 4 alternative metrics correctly distinguish empathy from random.** Cosine between separately-trained probes reflects classifier geometry, not concept structure; proper metrics work.

---

## Cycle 2: Length Validation & Confound Check

### Question
Do metrics work on trivially different features (length)? Is length confounded with empathy?

### Council Deliberation
- **PI**: Proposed testing length as gold-standard control
- **Statistician**: Required chi-square test for empathy-length confound
- **Engineer**: Suggested percentile binning (short/medium/long)
- **Devil's Advocate**: "If empathy correlates with length, we have a problem."
- **Consensus**: Green light with mandatory confound analysis

### Experiment
1. Binned responses by length (33rd/67th percentiles)
2. Computed all 4 metrics for length labels
3. Chi-square test for empathy-length association

### Results

| Metric | Empathy | Length | Random |
|--------|---------|--------|--------|
| d' | **12.1** | 1.77 | 1.0 |
| Probe Agree | **0.96** | 0.79 | 0.70 |
| Purity | **0.97** | 0.51 | 0.42 |
| AUROC | **1.00** | 0.79 | 0.43 |

**Confound Analysis:**
- Chi-square: 26.87
- p-value: 2.1e-05
- Cognitive responses tend longer (379 chars mean)
- Affective responses tend shorter (315 chars mean)

### Finding
**Significant confound detected** (p < 0.0001). However, Empathy >> Length >> Random on all metrics. Need to residualize to confirm empathy isn't just length.

---

## Cycle 3: Length-Residualized Empathy Analysis

### Question
Does empathy structure survive after removing length information from activations?

### Council Deliberation
- **PI**: Proposed linear regression to remove length from activations
- **Statistician**: Required R² to measure how much variance length explains
- **Engineer**: Suggested per-dimension regression for clean residualization
- **Devil's Advocate**: "What if 90% of the signal IS length? We'd lose everything."
- **Consensus**: Green light with retention metric (% of original structure preserved)

### Experiment
1. Regressed each activation dimension on response length
2. Used residuals as "length-free" activations
3. Recomputed all 4 metrics on residualized data
4. Compared to random baseline on residualized data

### Results

| Metric | Original | Residualized | % Retained |
|--------|----------|--------------|------------|
| d' | 12.1 | **11.0** | 91% |
| Probe Agree | 0.96 | **0.86** | 90% |
| Purity | 0.97 | **0.84** | 87% |
| AUROC | 1.00 | **0.96** | 96% |

**Variance Explained by Length:**
- Mean R²: 4.7%
- Max R²: 39%
- **Length explains less than 5% of activation variance on average**

**Comparison to Random:**

| Metric | Residualized Empathy | Residualized Random |
|--------|---------------------|---------------------|
| d' | **11.0** | 1.04 |
| Probe Agree | **0.86** | 0.70 |
| Purity | **0.84** | 0.39 |
| AUROC | **0.96** | 0.52 |

### Finding
**EMPATHY STRUCTURE IS REAL.** After removing length:
- 91% of empathy structure retained (average across metrics)
- Empathy still massively beats random on all metrics
- Length explains only 4.7% of activation variance

---

## Final Conclusions

### What We Established

1. **Cosine similarity between separately-trained probes reflects classifier geometry** - It doesn't measure concept structure in this use case
2. **Four alternative metrics all work** - d-prime, probe agreement, clustering purity, AUROC
3. **Length is a confound but not the explanation** - Only 4.7% variance, 91% structure survives
4. **Empathy structure is REAL** - Not a length artifact, survives rigorous validation

### Metric Recommendations

| Metric | Use For | Interpretation |
|--------|---------|----------------|
| AUROC | Classification accuracy | >0.9 = strong separability |
| d-prime | Effect size | >2 = meaningful separation |
| Probe Agreement | Cross-validation | >0.8 = stable probes |
| Clustering Purity | Natural grouping | >0.8 = clear clusters |
| Cosine | **DO NOT USE** | Reflects classifier geometry, not concept structure |

### Research Question Answered

> **Do LLMs represent empathy subtypes distinctly?**

**YES.** Cognitive, Affective, and Instrumental empathy occupy distinct regions in activation space:
- AUROC = 1.0 (perfect classification)
- d-prime = 12.1 (massive effect size)
- Clustering purity = 0.97 (near-perfect natural grouping)
- Structure survives length residualization (91% retention)

---

## Methodology Contribution

This study contributes a **methodological warning and fix**:

> **Warning:** Cosine similarity between linear probe weight vectors is not a valid metric for measuring concept-specific neural structure. Studies claiming concept decomposition based on probe cosines should be re-evaluated.

> **Fix:** Use cross-validated AUROC, d-prime, probe agreement, or clustering purity instead. These metrics correctly distinguish meaningful structure from random baselines.

---

## Data Files

| File | Description |
|------|-------------|
| `alternative_metrics.json` | Cycle 1: 4 metrics comparison |
| `length_validation.json` | Cycle 2: Length control + confound |
| `residualized_empathy.json` | Cycle 3: Length residualization |
| `null_distribution_mistral_100.json` | Original null test (100 perms) |
| `auroc_vs_cosine.json` | AUROC vs cosine comparison |

---

## Council Participants

- **Principal Investigator:** Research direction and experimental design
- **Statistician:** Methodology rigor, proper baselines, confound analysis
- **Engineer:** Implementation feasibility and metric selection
- **Devil's Advocate:** Assumption challenging and worst-case scenarios

---

## Timeline

| Cycle | Duration | Compute Cost |
|-------|----------|--------------|
| Cycle 1 | ~10 min | ~$0.50 |
| Cycle 2 | ~10 min | ~$0.50 |
| Cycle 3 | ~10 min | ~$0.50 |
| **Total** | ~30 min | ~$1.50 |

---

*Report generated: January 29, 2026*
*Model: Mistral-7B-Instruct-v0.3*
*GPU: NVIDIA RTX A5000 (RunPod)*
