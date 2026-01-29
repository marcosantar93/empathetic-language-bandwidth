#!/usr/bin/env python3
"""
Null Distribution Test: Compare empathy cosines to random response type cosines.
Tests whether empathy subtypes have statistically different separation than random triplets.
"""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import random

def load_model(model_name, device='cuda'):
    """Load model with TransformerLens."""
    from transformer_lens import HookedTransformer

    model_map = {
        'mistral-7b': 'mistralai/Mistral-7B-v0.1',
        'llama-3-8b': 'meta-llama/Meta-Llama-3-8B',
        'gemma-7b': 'google/gemma-7b',
        'gpt2-xl': 'gpt2-xl',
        'pythia-1.4b': 'EleutherAI/pythia-1.4b',
    }

    hf_name = model_map.get(model_name, model_name)
    model = HookedTransformer.from_pretrained(
        hf_name,
        device=device,
        torch_dtype=torch.float16
    )
    return model

def extract_activations(model, prompts, layer, device='cuda'):
    """Extract activations at specified layer."""
    activations = []

    for prompt in tqdm(prompts, desc=f"Extracting layer {layer}"):
        tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=f'blocks.{layer}.hook_resid_post')
        act = cache[f'blocks.{layer}.hook_resid_post'][0, -1, :].cpu().numpy()
        activations.append(act)

    return np.array(activations)

def train_probe(X, y):
    """Train logistic regression probe."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    return clf

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_triplet_cosines(acts, labels, label_names):
    """Compute pairwise cosines for a triplet of labels."""
    dirs = {}
    for label in label_names:
        mask = np.array([l == label for l in labels])
        if mask.sum() > 0:
            probe = train_probe(acts, mask.astype(int))
            dirs[label] = probe.coef_[0]

    cosines = {}
    pairs = [(label_names[0], label_names[1]),
             (label_names[0], label_names[2]),
             (label_names[1], label_names[2])]

    for l1, l2 in pairs:
        if l1 in dirs and l2 in dirs:
            cosines[f'{l1}_{l2}'] = cosine_similarity(dirs[l1], dirs[l2])

    return cosines

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mistral-7b')
    parser.add_argument('--data-dir', default='experiments/tripartite/data')
    parser.add_argument('--output', default='experiments/tripartite/results/null_distribution.json')
    parser.add_argument('--layer', type=int, default=16)
    parser.add_argument('--n-random', type=int, default=20, help='Number of random triplets to test')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    data_dir = Path(args.data_dir)

    # Load empathy data
    with open(data_dir / 'triplets_filtered.json') as f:
        empathy_data = json.load(f)['triplets']

    # Load control data
    with open(data_dir / 'controls_non_empathy.json') as f:
        control_data = json.load(f)['items']

    print(f"\nLoading model...")
    model = load_model(args.model, device)

    # 1. Compute empathy cosines
    print(f"\n{'='*60}")
    print("COMPUTING EMPATHY COSINES")
    print(f"{'='*60}")

    empathy_prompts, empathy_labels = [], []
    for item in empathy_data[:30]:
        for t in ['cognitive', 'affective', 'instrumental']:
            if t in item:
                empathy_prompts.append(f"{item['scenario']}\n\nResponse: {item[t]}")
                empathy_labels.append(t)

    empathy_acts = extract_activations(model, empathy_prompts, args.layer, device)
    empathy_cosines = compute_triplet_cosines(
        empathy_acts, empathy_labels,
        ['cognitive', 'affective', 'instrumental']
    )

    print(f"\nEmpathy cosines:")
    for k, v in empathy_cosines.items():
        print(f"  {k}: {v:.3f}")
    empathy_mean = np.mean(list(empathy_cosines.values()))
    print(f"  Mean: {empathy_mean:.3f}")

    # 2. Compute control cosines (non-empathy responses)
    print(f"\n{'='*60}")
    print("COMPUTING CONTROL COSINES")
    print(f"{'='*60}")

    control_prompts, control_labels = [], []
    for item in control_data[:30]:
        for t in ['response_1', 'response_2', 'response_3']:
            if t in item:
                control_prompts.append(f"{item['scenario']}\n\nResponse: {item[t]}")
                control_labels.append(t)

    control_acts = extract_activations(model, control_prompts, args.layer, device)
    control_cosines = compute_triplet_cosines(
        control_acts, control_labels,
        ['response_1', 'response_2', 'response_3']
    )

    print(f"\nControl cosines:")
    for k, v in control_cosines.items():
        print(f"  {k}: {v:.3f}")
    control_mean = np.mean(list(control_cosines.values()))
    print(f"  Mean: {control_mean:.3f}")

    # 3. Generate null distribution by random label shuffling
    print(f"\n{'='*60}")
    print(f"GENERATING NULL DISTRIBUTION ({args.n_random} permutations)")
    print(f"{'='*60}")

    null_cosines = []
    all_acts = empathy_acts  # Use empathy activations but shuffle labels

    for i in tqdm(range(args.n_random), desc="Random permutations"):
        # Create random labels (3 groups)
        n = len(all_acts)
        random_labels = []
        for j in range(n):
            random_labels.append(f'group_{j % 3}')
        random.shuffle(random_labels)

        # Compute cosines for random grouping
        rand_cosines = compute_triplet_cosines(
            all_acts, random_labels,
            ['group_0', 'group_1', 'group_2']
        )
        null_cosines.append(np.mean(list(rand_cosines.values())))

    null_mean = np.mean(null_cosines)
    null_std = np.std(null_cosines)

    print(f"\nNull distribution:")
    print(f"  Mean: {null_mean:.3f}")
    print(f"  Std: {null_std:.3f}")
    print(f"  Min: {min(null_cosines):.3f}")
    print(f"  Max: {max(null_cosines):.3f}")

    # 4. Statistical test
    print(f"\n{'='*60}")
    print("STATISTICAL COMPARISON")
    print(f"{'='*60}")

    # Z-score: how many SDs is empathy from null?
    z_empathy = (empathy_mean - null_mean) / null_std
    z_control = (control_mean - null_mean) / null_std

    # Percentile
    empathy_percentile = np.mean([c > empathy_mean for c in null_cosines]) * 100
    control_percentile = np.mean([c > control_mean for c in null_cosines]) * 100

    print(f"\nEmpathy vs Null:")
    print(f"  Empathy mean: {empathy_mean:.3f}")
    print(f"  Null mean: {null_mean:.3f}")
    print(f"  Z-score: {z_empathy:.2f}")
    print(f"  Percentile: {empathy_percentile:.1f}% of null > empathy")

    print(f"\nControl vs Null:")
    print(f"  Control mean: {control_mean:.3f}")
    print(f"  Null mean: {null_mean:.3f}")
    print(f"  Z-score: {z_control:.2f}")
    print(f"  Percentile: {control_percentile:.1f}% of null > control")

    print(f"\nEmpathy vs Control:")
    print(f"  Difference: {empathy_mean - control_mean:.3f}")
    print(f"  Empathy more negative: {empathy_mean < control_mean}")

    # Conclusion
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")

    if z_empathy < -2:
        conclusion = "EMPATHY IS SIGNIFICANTLY MORE SEPARATED than random"
    elif z_empathy > 2:
        conclusion = "EMPATHY IS SIGNIFICANTLY LESS SEPARATED than random"
    else:
        conclusion = "EMPATHY IS NOT SIGNIFICANTLY DIFFERENT from random"

    print(f"\n{conclusion}")
    print(f"\nInterpretation:")
    if abs(empathy_mean - control_mean) < 0.05:
        print("  - Empathy and control cosines are nearly identical")
        print("  - Separation is NOT empathy-specific")
    else:
        print(f"  - Empathy-control difference: {empathy_mean - control_mean:.3f}")

    # Save results
    results = {
        'model': args.model,
        'layer': args.layer,
        'empathy_cosines': empathy_cosines,
        'empathy_mean': empathy_mean,
        'control_cosines': control_cosines,
        'control_mean': control_mean,
        'null_distribution': {
            'mean': null_mean,
            'std': null_std,
            'min': min(null_cosines),
            'max': max(null_cosines),
            'values': null_cosines
        },
        'statistics': {
            'z_empathy': z_empathy,
            'z_control': z_control,
            'empathy_percentile': empathy_percentile,
            'control_percentile': control_percentile,
            'empathy_control_diff': empathy_mean - control_mean
        },
        'conclusion': conclusion
    }

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")

if __name__ == '__main__':
    main()
