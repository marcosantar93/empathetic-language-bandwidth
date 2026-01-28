#!/usr/bin/env python3
"""
Run control analysis: Compare empathy triplets with non-empathy emotional responses.
Shows that tripartite structure is specific to empathy.
"""

import json
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from extract_activations import load_model, extract_activations
from train_probes import train_probe, compute_direction_cosines

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen2.5-7b')
    parser.add_argument('--data-dir', default='experiments/tripartite/data')
    parser.add_argument('--output', default='experiments/tripartite/results/control_analysis.json')
    args = parser.parse_args()

    print("="*60)
    print("CONTROL ANALYSIS: Empathy vs Non-Empathy Structure")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load datasets
    data_dir = Path(args.data_dir)

    with open(data_dir / 'triplets_filtered.json') as f:
        empathy_data = json.load(f)

    with open(data_dir / 'controls_non_empathy.json') as f:
        control_data = json.load(f)

    print(f"Empathy samples: {len(empathy_data)}")
    print(f"Control samples: {len(control_data)}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model(args.model, device)

    # Determine layer
    layer = 14 if 'qwen' in args.model.lower() else 16

    # Extract empathy activations
    print("\nExtracting empathy activations...")
    empathy_prompts = []
    empathy_labels = []
    for item in empathy_data[:30]:  # Use subset for speed
        for resp_type in ['cognitive', 'affective', 'instrumental']:
            if resp_type in item:
                prompt = f"{item['scenario']}\n\nResponse: {item[resp_type]}"
                empathy_prompts.append(prompt)
                empathy_labels.append(resp_type)

    empathy_acts = extract_activations(model, tokenizer, empathy_prompts, layer, device)

    # Extract control activations
    print("Extracting control activations...")
    control_prompts = []
    control_labels = []
    for item in control_data[:30]:
        for resp_type in ['response_1', 'response_2', 'response_3']:
            if resp_type in item:
                prompt = f"{item['scenario']}\n\nResponse: {item[resp_type]}"
                control_prompts.append(prompt)
                control_labels.append(resp_type)

    control_acts = extract_activations(model, tokenizer, control_prompts, layer, device)

    # Train probes for empathy
    print("\nTraining empathy probes...")
    empathy_directions = {}
    for label in ['cognitive', 'affective', 'instrumental']:
        mask = np.array([l == label for l in empathy_labels])
        if mask.sum() > 0:
            probe = train_probe(empathy_acts, mask)
            empathy_directions[label] = probe.coef_[0]

    # Train probes for control
    print("Training control probes...")
    control_directions = {}
    for label in ['response_1', 'response_2', 'response_3']:
        mask = np.array([l == label for l in control_labels])
        if mask.sum() > 0:
            probe = train_probe(control_acts, mask)
            control_directions[label] = probe.coef_[0]

    # Compute cosines
    print("\nComputing cosine similarities...")

    empathy_cosines = compute_direction_cosines(empathy_directions)
    control_cosines = compute_direction_cosines(control_directions)

    results = {
        'model': args.model,
        'layer': layer,
        'empathy': {
            'n_samples': len(empathy_prompts),
            'cosines': empathy_cosines
        },
        'control': {
            'n_samples': len(control_prompts),
            'cosines': control_cosines
        },
        'comparison': {
            'empathy_mean_separation': np.mean(list(empathy_cosines.values())),
            'control_mean_separation': np.mean(list(control_cosines.values())),
            'specificity_ratio': abs(np.mean(list(empathy_cosines.values()))) / max(abs(np.mean(list(control_cosines.values()))), 0.01)
        }
    }

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nEmpathy cosines:")
    for k, v in empathy_cosines.items():
        print(f"  {k}: {v:.3f}")
    print(f"  Mean: {results['comparison']['empathy_mean_separation']:.3f}")

    print(f"\nControl cosines:")
    for k, v in control_cosines.items():
        print(f"  {k}: {v:.3f}")
    print(f"  Mean: {results['comparison']['control_mean_separation']:.3f}")

    print(f"\nSpecificity ratio: {results['comparison']['specificity_ratio']:.2f}x")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")

if __name__ == '__main__':
    main()
