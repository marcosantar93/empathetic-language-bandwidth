# Session Notes: Empathy Structure Research

**Last Updated:** January 31, 2026
**Status:** Ready to share / Publication ready

---

## What We Accomplished

### Phase 2 Research Complete

Investigated whether empathy decomposes into distinct subspaces (Cognitive/Affective/Instrumental) in LLM activation spaces.

**Key Findings:**
1. Empathy structure is real and robust (AUROC = 0.98-1.0 across 4 models)
2. All 3 empathy types simultaneously distinguishable (89.3% accuracy vs 33% chance)
3. Empathy is independent of formality and general emotion (orthogonal subspaces)
4. Empathy directions are causally meaningful (70%+ probability shifts via intervention)
5. Structure emerges at Layer 1 and is uniform across token positions

### Methodology Discovery

Discovered that **cosine similarity between separately-trained probe weight vectors** doesn't measure concept structure—it reflects classifier geometry. This is specific to comparing weights of separately-trained binary classifiers.

**Proper metrics identified:** AUROC, d-prime, clustering purity

### Documentation Refined

Updated all reports to:
- Scope the cosine claim precisely (not "broken", but "inappropriate for this use case")
- Add theoretical context (read/write distinction, feature superposition)
- Add proper citations (Steck 2024, Park 2023, Gorton 2024, Elhage 2022, Wehner 2025)
- Include limitations and scope section

---

## Current Repository State

```
empathetic-language-bandwidth/
├── README.md                    # Updated with Phase 2 reproduction instructions
├── experiments/tripartite/
│   ├── FINAL_REPORT.md          # Full technical report (40KB) - PUBLICATION READY
│   ├── BLOG_POST.md             # Public summary - PUBLICATION READY
│   ├── COUNCIL_REPORT*.md       # 4 rounds of methodology validation
│   ├── data/                    # Triplets and controls (included)
│   ├── results/                 # 20+ JSON files with pre-computed results
│   └── scripts/                 # Runnable experiment scripts
```

**Git status:** Clean, synced with remote (commit 6d4a69c)

---

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `FINAL_REPORT.md` | Full technical writeup | Ready |
| `BLOG_POST.md` | Public blog post | Ready |
| `COUNCIL_REPORT.md` | Round 1-3 methodology | Complete |
| `COUNCIL_REPORT_ROUND4.md` | Causal intervention tests | Complete |
| `data/triplets_filtered.json` | 90 scenarios × 3 empathy types | Included |
| `results/*.json` | All experiment results | Included |

---

## Decisions Made

1. **Scoped the cosine claim** - Changed from "cosine similarity is broken" to "cosine similarity between separately-trained probe weights reflects classifier geometry, not concept structure"

2. **Added theoretical grounding** - Read/write distinction (Gorton), feature superposition (Elhage), cosine issues (Steck, Park)

3. **Kept scope narrow** - Finding applies specifically to comparing separately-trained binary classifiers; cosine remains valid for other RepE uses

4. **Included limitations** - Sample size (270 triplets), model range (1.1B-7B), English only, no human evaluation

---

## What's NOT Done (Future Work)

From `docs/FUTURE_EXPERIMENTS.md` and README:

1. **Generation-Time Steering** - Apply empathy directions during actual text generation
2. **Human Evaluation** - Correlate geometric measures with human-rated empathy
3. **Scaling to 70B+** - Confirm scale independence at larger sizes
4. **Base vs Instruct** - Compare empathy structure in base models
5. **Cross-lingual** - Test if empathy structure exists in non-English models
6. **Other Concepts** - Apply methodology to humor, persuasion, creativity

---

## To Resume Work

### If continuing experiments:
```bash
cd empathetic-language-bandwidth
source venv/bin/activate

# Run validation on new model
cd experiments/tripartite
python scripts/run_all_validation.py --model <model_name>
```

### If publishing blog:
- Blog post ready at: `experiments/tripartite/BLOG_POST.md`
- Website repo location: Ask user (different project)
- Prompt for updating website was provided in previous session

### If updating repo:
- All changes committed and pushed
- Remote: github.com/marcosantar93/empathetic-language-bandwidth
- SSH was timing out; used HTTPS for last push

---

## RunPod Status

All pods terminated (completed experiments):
- 7daoedfmrnuwbh (safety-exp-v3)
- 8wiorumeu619p2 (empathy-round4)
- l07123ixjezu7x

---

## Council Process Used

Each experiment cycle followed:
1. **Proposal** - PI proposes experiment
2. **Review** - Statistician, Engineer, Devil's Advocate critique
3. **Consensus** - Green light when concerns addressed
4. **Execution** - Run on GPU
5. **Analysis** - Interpret and plan next

This process caught the cosine similarity issue and led to proper validation.

---

## Contact / Links

- **GitHub:** https://github.com/marcosantar93/empathetic-language-bandwidth
- **Blog:** https://marcosantar.com/blog/empathetic-language-bandwidth
- **Author:** Marco Santarcangelo

---

*Notes written: January 31, 2026*
