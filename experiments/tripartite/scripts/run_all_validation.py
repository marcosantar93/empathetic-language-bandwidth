#!/usr/bin/env python3
"""
Run all validation experiments: Control, Multi-layer, AUROC
Self-contained script that doesn't depend on other modules.
"""

import json
import os
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Model loading
def load_model(model_name, device='cuda'):
    """Load model with TransformerLens."""
    from transformer_lens import HookedTransformer

    model_map = {
        'qwen2.5-7b': 'Qwen/Qwen2.5-7B',
        'mistral-7b': 'mistralai/Mistral-7B-v0.1',
        'llama-3-8b': 'meta-llama/Meta-Llama-3-8B',
        'llama-3.1-8b': 'meta-llama/Llama-3.1-8B',
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
        tokens = model.to_tokens(prompt, prepend_bos=True)
        tokens = tokens.to(device)

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

def run_control_analysis(model, data_dir, layer):
    """Compare empathy vs non-empathy structure."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Control Analysis")
    print("="*60)

    with open(data_dir / 'triplets_filtered.json') as f:
        empathy_data = json.load(f)
    with open(data_dir / 'controls_non_empathy.json') as f:
        control_data = json.load(f)

    # Prepare empathy prompts
    empathy_prompts, empathy_labels = [], []
    for item in empathy_data[:30]:
        for t in ['cognitive', 'affective', 'instrumental']:
            if t in item:
                empathy_prompts.append(f"{item['scenario']}\n\nResponse: {item[t]}")
                empathy_labels.append(t)

    # Prepare control prompts
    control_prompts, control_labels = [], []
    for item in control_data[:30]:
        for t in ['response_1', 'response_2', 'response_3']:
            if t in item:
                control_prompts.append(f"{item['scenario']}\n\nResponse: {item[t]}")
                control_labels.append(t)

    # Extract activations
    print("Extracting empathy activations...")
    empathy_acts = extract_activations(model, empathy_prompts, layer)
    print("Extracting control activations...")
    control_acts = extract_activations(model, control_prompts, layer)

    # Train probes and get directions
    empathy_dirs = {}
    for label in ['cognitive', 'affective', 'instrumental']:
        mask = np.array([l == label for l in empathy_labels])
        if mask.sum() > 0:
            probe = train_probe(empathy_acts, mask.astype(int))
            empathy_dirs[label] = probe.coef_[0]

    control_dirs = {}
    for label in ['response_1', 'response_2', 'response_3']:
        mask = np.array([l == label for l in control_labels])
        if mask.sum() > 0:
            probe = train_probe(control_acts, mask.astype(int))
            control_dirs[label] = probe.coef_[0]

    # Compute cosines
    empathy_cos = {
        'cog_aff': cosine_similarity(empathy_dirs['cognitive'], empathy_dirs['affective']),
        'cog_instr': cosine_similarity(empathy_dirs['cognitive'], empathy_dirs['instrumental']),
        'aff_instr': cosine_similarity(empathy_dirs['affective'], empathy_dirs['instrumental']),
    }

    control_cos = {
        'r1_r2': cosine_similarity(control_dirs['response_1'], control_dirs['response_2']),
        'r1_r3': cosine_similarity(control_dirs['response_1'], control_dirs['response_3']),
        'r2_r3': cosine_similarity(control_dirs['response_2'], control_dirs['response_3']),
    }

    print(f"\nEmpathy cosines: {empathy_cos}")
    print(f"Control cosines: {control_cos}")
    print(f"Empathy mean: {np.mean(list(empathy_cos.values())):.3f}")
    print(f"Control mean: {np.mean(list(control_cos.values())):.3f}")

    return {'empathy': empathy_cos, 'control': control_cos}

def run_multilayer_sweep(model, data_dir, layers):
    """Check separation across layers."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Multi-layer Sweep")
    print("="*60)

    with open(data_dir / 'triplets_filtered.json') as f:
        data = json.load(f)

    prompts, labels = [], []
    for item in data[:30]:
        for t in ['cognitive', 'affective', 'instrumental']:
            if t in item:
                prompts.append(f"{item['scenario']}\n\nResponse: {item[t]}")
                labels.append(t)

    results = {}
    for layer in layers:
        print(f"\nLayer {layer}:")
        acts = extract_activations(model, prompts, layer)

        dirs = {}
        for label in ['cognitive', 'affective', 'instrumental']:
            mask = np.array([l == label for l in labels])
            if mask.sum() > 0:
                probe = train_probe(acts, mask.astype(int))
                dirs[label] = probe.coef_[0]

        cog_aff = cosine_similarity(dirs['cognitive'], dirs['affective'])
        results[layer] = cog_aff
        print(f"  cos(Cog,Aff): {cog_aff:.3f}")

    print(f"\nBest layer: {min(results, key=results.get)} (cos={min(results.values()):.3f})")
    return results

def run_auroc_analysis(model, data_dir, layer):
    """Compute AUROC for binary classifications."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: AUROC Analysis")
    print("="*60)

    with open(data_dir / 'triplets_filtered.json') as f:
        data = json.load(f)

    prompts, labels = [], []
    for item in data[:60]:
        for t in ['cognitive', 'affective', 'instrumental']:
            if t in item:
                prompts.append(f"{item['scenario']}\n\nResponse: {item[t]}")
                labels.append(t)

    print("Extracting activations...")
    acts = extract_activations(model, prompts, layer)
    labels_arr = np.array(labels)

    pairs = [
        ('cognitive', 'affective'),
        ('cognitive', 'instrumental'),
        ('affective', 'instrumental')
    ]

    results = {}
    for l1, l2 in pairs:
        mask = (labels_arr == l1) | (labels_arr == l2)
        X = acts[mask]
        y = (labels_arr[mask] == l1).astype(int)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        try:
            scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
            auroc = scores.mean()
        except:
            clf.fit(X, y)
            auroc = roc_auc_score(y, clf.predict_proba(X)[:, 1])

        results[f'{l1}_vs_{l2}'] = auroc
        print(f"  {l1} vs {l2}: AUROC = {auroc:.3f}")

    print(f"\nMean AUROC: {np.mean(list(results.values())):.3f}")
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen2.5-7b')
    parser.add_argument('--data-dir', default='experiments/tripartite/data')
    parser.add_argument('--output', default='experiments/tripartite/results/validation_results.json')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    data_dir = Path(args.data_dir)

    # Load model once
    print("\nLoading model...")
    model = load_model(args.model, device)

    n_layers = model.cfg.n_layers
    default_layer = 14 if 'qwen' in args.model.lower() else 16
    layers_to_test = [4, 8, 12, default_layer, 20, min(24, n_layers-1)]
    layers_to_test = sorted(set([l for l in layers_to_test if l < n_layers]))

    # Run all experiments
    results = {
        'model': args.model,
        'device': device,
        'default_layer': default_layer
    }

    results['control'] = run_control_analysis(model, data_dir, default_layer)
    results['multilayer'] = run_multilayer_sweep(model, data_dir, layers_to_test)
    results['auroc'] = run_auroc_analysis(model, data_dir, default_layer)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n1. Control Analysis:")
    print(f"   Empathy mean separation: {np.mean(list(results['control']['empathy'].values())):.3f}")
    print(f"   Control mean separation: {np.mean(list(results['control']['control'].values())):.3f}")

    print(f"\n2. Multi-layer Sweep:")
    best_layer = min(results['multilayer'], key=results['multilayer'].get)
    print(f"   Best layer: {best_layer} (cos={results['multilayer'][best_layer]:.3f})")

    print(f"\n3. AUROC Analysis:")
    print(f"   Mean AUROC: {np.mean(list(results['auroc'].values())):.3f}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")

if __name__ == '__main__':
    main()
