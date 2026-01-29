# We Tried to Measure Empathy in LLMs. We Found a Flaw in the Entire Methodology.

*How a failed experiment revealed that a common interpretability metric doesn't measure what we think it does.*

---

## The Original Question

Do large language models represent empathy as a single concept, or do they decompose it into distinct subtypes?

Psychologists have long distinguished between **cognitive empathy** (understanding someone's perspective), **affective empathy** (sharing their feelings), and **instrumental empathy** (offering practical help). We wanted to know: do LLMs encode these as separate "directions" in their activation space?

This matters for AI safety. If we can identify where empathy "lives" in a model, we might be able to steer it—making AI assistants more compassionate, or understanding when they're being manipulative.

## The Standard Approach

We followed the representation engineering playbook:

1. Generate scenarios with matched Cognitive, Affective, and Instrumental responses
2. Extract activations from multiple LLMs (Mistral-7B, Llama-3-8B, etc.)
3. Train linear probes to classify each empathy type
4. Measure cosine similarity between the probe weight vectors
5. If cosine < 0.5, the concepts are "distinct"

Our initial results looked promising: **cos(Cognitive, Affective) = -0.29**. Negative cosine—the directions point in opposite directions! The empathy subtypes appear to occupy distinct subspaces!

We were ready to write up the paper.

## The First Red Flag

Then we ran a control experiment. What if we used non-empathy responses—just three arbitrary response styles per scenario?

| Condition | Mean Cosine |
|-----------|-------------|
| Empathy (Cog/Aff/Instr) | -0.484 |
| Control (arbitrary types) | -0.490 |

**Nearly identical.** The "separation" we found wasn't specific to empathy at all. Any set of distinct response types showed the same pattern.

## The Null Distribution Test

Maybe both empathy and controls have meaningful structure? We needed a proper baseline: random label permutations.

We shuffled the labels randomly 100 times and computed cosines for each permutation. If empathy has real structure, it should show MORE separation (more negative cosines) than random.

**Result: Z = +12.9**

The Z-score was *positive*. Empathy labels produced LESS separation than random shuffling. Not "not significant"—actively worse than chance.

## The Council Process

At this point, we convened a research council—multiple perspectives to stress-test the findings:

**Principal Investigator**: "Is this specific to empathy, or is the methodology broken?"

**Statistician**: "We need a gold-standard control. Something trivially separable."

**Engineer**: "What about response length? It's computable from the text itself."

**Devil's Advocate**: "If length fails too, we've learned something bigger than empathy."

## Cycle 2: The Length Test

We binned responses by character length:
- Short: mean 300 chars
- Medium: mean 346 chars
- Long: mean 396 chars

Clearly different. Trivially separable. If our methodology works, length should beat random.

**Result: Z = +11.6**

Length ALSO showed a positive Z-score. Even a trivially different feature—one you can compute with `len()`—failed the cosine test.

## Cycle 3: The Breakthrough

The statistician proposed the key experiment: what if we measure classification accuracy (AUROC) instead of cosine similarity?

| Feature | AUROC | Cosine Z-score |
|---------|-------|----------------|
| Length | **0.963** | +11.6 |
| Empathy | **1.000** | +18.0 |

**The probes achieve near-perfect classification.** AUROC of 0.96-1.0 means the linear probes CAN find the structure in the data. They successfully distinguish cognitive from affective empathy, and short from long responses.

But the cosine metric says they're WORSE than random.

**The probes work. The metric doesn't.**

## Why Cosines Fail

Here's the geometry: binary logistic regression finds a hyperplane that separates two classes. The weight vector points toward the positive class.

When you train separate probes for different concepts, each probe's weights point toward its respective positive class. These directions are naturally different—that's the whole point. The resulting cosines are negative because the probes are solving different classification problems.

Random label permutations maximize this effect because they create maximally distinct (if meaningless) classification targets. That's why random labels produce the MOST negative cosines.

**The negative cosines reflect classifier geometry, not concept structure.**

## What This Means

### For Representation Engineering

A lot of papers use probe cosine similarity as evidence for concept decomposition:
- "Truthfulness and sycophancy occupy distinct subspaces (cos = -0.3)"
- "Safety and helpfulness directions are orthogonal"
- "Emotions cluster by valence in activation space"

These claims may need re-evaluation. The cosine metric doesn't measure what we thought it did.

### For Our Empathy Study

Ironically, empathy subtypes ARE linearly separable (AUROC = 1.0). Models DO represent cognitive and affective empathy differently—you can train a perfect classifier to distinguish them.

But the cosine similarity between probe directions tells us nothing about whether this is "empathy-specific" structure or just "these responses are different" structure.

### For AI Safety

If you're using probe directions for steering or monitoring, the directions themselves may be valid (they classify correctly). But don't interpret the cosine between different probes as measuring anything meaningful about concept relationships.

## The Methodological Fix

Use proper metrics:

1. **AUROC** for classification accuracy (do probes find the structure?)
2. **Null distribution testing** for statistical validity (is it better than random?)
3. **Control conditions** for specificity (is it unique to this concept?)

Don't rely on cosine similarity between probe vectors. It's an artifact of classifier geometry.

## Conclusion

We set out to measure empathy decomposition in LLMs. We ended up discovering a flaw in how the field measures concept structure.

The lesson: when a metric gives you the answer you expect, that's exactly when you should stress-test it hardest. We expected negative cosines to mean "distinct concepts." We were wrong.

Science often works this way. The failed experiment teaches you more than the successful one—if you're willing to follow the thread.

---

*Code and data: [github.com/marcosantar93/empathetic-language-bandwidth](https://github.com/marcosantar93/empathetic-language-bandwidth)*

*Full technical report: See FINAL_REPORT.md in the repository*

---

**TL;DR**: We tried to measure whether LLMs decompose empathy into subtypes. We discovered that the standard metric (cosine similarity between probes) is fundamentally broken—it reflects classifier geometry, not concept structure. Probes with perfect classification accuracy (AUROC=1.0) show WORSE than random on the cosine metric. The probes work; the metric doesn't.
