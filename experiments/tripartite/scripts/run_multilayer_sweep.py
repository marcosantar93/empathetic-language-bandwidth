#!/usr/bin/env python3
"""
Multi-layer sweep: Check how Cog-Aff separation changes across layers.
Shows where tripartite structure emerges.
"""

import json
import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extract_activations import load_model, extract_activations
from train_probes import train_probe, compute_direction_cosines

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen2.5-7b')
    parser.add_argument('--data-dir', default='experiments/tripartite/data')
    parser.add_argument('--output', default='experiments/tripartite/results/multilayer_sweep.json')
    args = parser.parse_args()

    print("="*60)
    print("MULTI-LAYER SWEEP: Tripartite Structure Emergence")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    with open(Path(args.data_dir) / 'triplets_filtered.json') as f:
        data = json.load(f)

    # Prepare prompts
    prompts = []
    labels = []
    for item in data[:30]:  # Subset for speed
        for resp_type in ['cognitive', 'affective', 'instrumental']:
            if resp_type in item:
                prompt = f"{item['scenario']}\n\nResponse: {item[resp_type]}"
                prompts.append(prompt)
                labels.append(resp_type)

    print(f"Samples: {len(prompts)}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model(args.model, device)

    n_layers = model.cfg.n_layers

    # Layers to test
    if 'qwen' in args.model.lower():
        layers = [4, 7, 10, 14, 18, 22]
    else:
        layers = [4, 8, 12, 16, 20, 24]

    layers = [l for l in layers if l < n_layers]

    print(f"Testing layers: {layers}")

    results = {
        'model': args.model,
        'n_layers': n_layers,
        'layers_tested': layers,
        'per_layer': {}
    }

    for layer in layers:
        print(f"\n--- Layer {layer} ---")

        # Extract activations
        acts = extract_activations(model, tokenizer, prompts, layer, device)

        # Train probes
        directions = {}
        for label in ['cognitive', 'affective', 'instrumental']:
            mask = np.array([l == label for l in labels])
            if mask.sum() > 0:
                probe = train_probe(acts, mask)
                directions[label] = probe.coef_[0]

        # Compute cosines
        cosines = compute_direction_cosines(directions)

        results['per_layer'][layer] = {
            'cosines': cosines,
            'cog_aff': cosines.get('cognitive_affective', 0),
            'mean_separation': np.mean(list(cosines.values()))
        }

        print(f"  cos(Cog,Aff): {cosines.get('cognitive_affective', 0):.3f}")
        print(f"  Mean separation: {np.mean(list(cosines.values())):.3f}")

    # Find best layer
    best_layer = min(results['per_layer'].keys(),
                     key=lambda l: results['per_layer'][l]['cog_aff'])
    results['best_layer'] = best_layer
    results['best_separation'] = results['per_layer'][best_layer]['cog_aff']

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nCog-Aff separation by layer:")
    for layer in layers:
        sep = results['per_layer'][layer]['cog_aff']
        bar = '█' * int(abs(sep) * 20)
        print(f"  Layer {layer:2d}: {sep:+.3f} {bar}")
    print(f"\nBest layer: {best_layer} (cos = {results['best_separation']:.3f})")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")

if __name__ == '__main__':
    main()
