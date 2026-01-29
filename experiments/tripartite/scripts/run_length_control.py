#!/usr/bin/env python3
"""
Length Control Test: Compare empathy cosines to response length cosines.
Tests whether our methodology can detect trivially separable features.
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

def get_length_categories(texts):
    """Categorize texts by percentile-based length bins."""
    lengths = [len(t) for t in texts]
    p33 = np.percentile(lengths, 33)
    p66 = np.percentile(lengths, 66)

    categories = []
    for length in lengths:
        if length <= p33:
            categories.append('short')
        elif length >= p66:
            categories.append('long')
        else:
            categories.append('medium')
    return categories

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
    parser.add_argument('--output', default='experiments/tripartite/results/length_control.json')
    parser.add_argument('--layer', type=int, default=16)
    parser.add_argument('--n-random', type=int, default=50)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    data_dir = Path(args.data_dir)

    # Load empathy data
    with open(data_dir / 'triplets_filtered.json') as f:
        empathy_data = json.load(f)['triplets']

    print(f"\nLoading model...")
    model = load_model(args.model, device)

    # Prepare all responses with empathy labels first
    all_prompts = []
    all_responses = []
    empathy_labels = []

    for item in empathy_data[:30]:
        for t in ['cognitive', 'affective', 'instrumental']:
            if t in item:
                response = item[t]
                prompt = f"{item['scenario']}\n\nResponse: {response}"
                all_prompts.append(prompt)
                all_responses.append(response)
                empathy_labels.append(t)

    # Compute length labels using percentile-based binning
    length_labels = get_length_categories(all_responses)

    # Show length distribution
    from collections import Counter
    print(f"\nLength distribution: {Counter(length_labels)}")
    print(f"Empathy distribution: {Counter(empathy_labels)}")

    # Extract activations once
    print(f"\n{'='*60}")
    print("EXTRACTING ACTIVATIONS")
    print(f"{'='*60}")

    all_acts = extract_activations(model, all_prompts, args.layer, device)

    # 1. Compute empathy cosines
    print(f"\n{'='*60}")
    print("COMPUTING EMPATHY COSINES")
    print(f"{'='*60}")

    empathy_cosines = compute_triplet_cosines(
        all_acts, empathy_labels,
        ['cognitive', 'affective', 'instrumental']
    )
    empathy_mean = np.mean(list(empathy_cosines.values()))

    print(f"\nEmpathy cosines:")
    for k, v in empathy_cosines.items():
        print(f"  {k}: {v:.3f}")
    print(f"  Mean: {empathy_mean:.3f}")

    # 2. Compute length cosines
    print(f"\n{'='*60}")
    print("COMPUTING LENGTH COSINES")
    print(f"{'='*60}")

    length_cosines = compute_triplet_cosines(
        all_acts, length_labels,
        ['short', 'medium', 'long']
    )
    length_mean = np.mean(list(length_cosines.values()))

    print(f"\nLength cosines:")
    for k, v in length_cosines.items():
        print(f"  {k}: {v:.3f}")
    print(f"  Mean: {length_mean:.3f}")

    # 3. Null distribution for both
    print(f"\n{'='*60}")
    print(f"GENERATING NULL DISTRIBUTION ({args.n_random} permutations)")
    print(f"{'='*60}")

    null_cosines = []
    for i in tqdm(range(args.n_random), desc="Random permutations"):
        n = len(all_acts)
        random_labels = [f'group_{j % 3}' for j in range(n)]
        random.shuffle(random_labels)

        rand_cosines = compute_triplet_cosines(
            all_acts, random_labels,
            ['group_0', 'group_1', 'group_2']
        )
        null_cosines.append(np.mean(list(rand_cosines.values())))

    null_mean = np.mean(null_cosines)
    null_std = np.std(null_cosines)

    # 4. Statistical comparison
    print(f"\n{'='*60}")
    print("STATISTICAL COMPARISON")
    print(f"{'='*60}")

    z_empathy = (empathy_mean - null_mean) / null_std
    z_length = (length_mean - null_mean) / null_std

    print(f"\nNull distribution: mean={null_mean:.3f}, std={null_std:.4f}")
    print(f"\nEmpathy: mean={empathy_mean:.3f}, Z={z_empathy:.2f}")
    print(f"Length:  mean={length_mean:.3f}, Z={z_length:.2f}")

    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")

    if z_length < -2 and z_empathy > -2:
        conclusion = "METHODOLOGY VALIDATED: Length separates, empathy doesn't"
        interpretation = "Probes work on trivial features; empathy genuinely lacks structure"
    elif z_length < -2 and z_empathy < -2:
        conclusion = "BOTH SEPARATE: Empathy may have structure after all"
        interpretation = "Need to revisit earlier conclusions"
    elif z_length > -2 and z_empathy > -2:
        conclusion = "NEITHER SEPARATES: Methodology may be flawed"
        interpretation = "Probes can't even find trivial features"
    else:
        conclusion = "UNEXPECTED: Length doesn't separate but empathy does"
        interpretation = "Very strange result, investigate further"

    print(f"\n{conclusion}")
    print(f"\nInterpretation: {interpretation}")
    print(f"\nLength more separable than empathy: {length_mean < empathy_mean}")

    # Save results
    results = {
        'model': args.model,
        'layer': args.layer,
        'empathy_cosines': empathy_cosines,
        'empathy_mean': empathy_mean,
        'length_cosines': length_cosines,
        'length_mean': length_mean,
        'null_distribution': {
            'mean': null_mean,
            'std': null_std,
            'values': null_cosines
        },
        'statistics': {
            'z_empathy': z_empathy,
            'z_length': z_length,
        },
        'conclusion': conclusion,
        'interpretation': interpretation
    }

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")

if __name__ == '__main__':
    main()
