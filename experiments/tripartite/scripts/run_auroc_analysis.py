#!/usr/bin/env python3
"""
Probe AUROC analysis: Train binary classifiers and report AUROC.
High AUROC = strong linear separability.
"""

import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from extract_activations import load_model, extract_activations
from train_probes import train_probe

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen2.5-7b')
    parser.add_argument('--data-dir', default='experiments/tripartite/data')
    parser.add_argument('--output', default='experiments/tripartite/results/auroc_analysis.json')
    args = parser.parse_args()

    print("="*60)
    print("AUROC ANALYSIS: Linear Separability of Empathy Subtypes")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    with open(Path(args.data_dir) / 'triplets_filtered.json') as f:
        data = json.load(f)

    # Prepare prompts
    prompts = []
    labels = []
    for item in data[:60]:  # More samples for better AUROC estimate
        for resp_type in ['cognitive', 'affective', 'instrumental']:
            if resp_type in item:
                prompt = f"{item['scenario']}\n\nResponse: {item[resp_type]}"
                prompts.append(prompt)
                labels.append(resp_type)

    print(f"Samples: {len(prompts)}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model(args.model, device)

    layer = 14 if 'qwen' in args.model.lower() else 16

    # Extract activations
    print(f"Extracting activations at layer {layer}...")
    acts = extract_activations(model, tokenizer, prompts, layer, device)
    labels_arr = np.array(labels)

    # Binary classification pairs
    pairs = [
        ('cognitive', 'affective'),
        ('cognitive', 'instrumental'),
        ('affective', 'instrumental')
    ]

    results = {
        'model': args.model,
        'layer': layer,
        'n_samples': len(prompts),
        'auroc': {}
    }

    print("\nTraining binary classifiers...")
    for label1, label2 in pairs:
        # Get samples for this pair
        mask = (labels_arr == label1) | (labels_arr == label2)
        X = acts[mask]
        y = (labels_arr[mask] == label1).astype(int)

        # Train with cross-validation
        clf = LogisticRegression(max_iter=1000, random_state=42)

        # Cross-validated AUROC
        try:
            scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
            auroc = scores.mean()
            auroc_std = scores.std()
        except:
            clf.fit(X, y)
            probs = clf.predict_proba(X)[:, 1]
            auroc = roc_auc_score(y, probs)
            auroc_std = 0

        pair_name = f"{label1}_vs_{label2}"
        results['auroc'][pair_name] = {
            'auroc': float(auroc),
            'std': float(auroc_std),
            'n_samples': int(mask.sum())
        }

        print(f"  {label1} vs {label2}: AUROC = {auroc:.3f} (+/- {auroc_std:.3f})")

    # Summary
    mean_auroc = np.mean([v['auroc'] for v in results['auroc'].values()])
    results['mean_auroc'] = float(mean_auroc)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nMean AUROC: {mean_auroc:.3f}")

    if mean_auroc > 0.8:
        print("Interpretation: STRONG linear separability")
    elif mean_auroc > 0.7:
        print("Interpretation: MODERATE linear separability")
    else:
        print("Interpretation: WEAK linear separability")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")

if __name__ == '__main__':
    main()
