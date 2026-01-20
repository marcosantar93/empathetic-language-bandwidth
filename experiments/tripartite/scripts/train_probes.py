#!/usr/bin/env python3
"""
Train linear probes for tripartite empathy classification.

Experiment B: Theory-driven validation
- Train probes to classify Cognitive vs Affective vs Instrumental empathy
- Measure linear separability (AUROC)
- Extract probe direction vectors for convergence analysis

Usage:
    python train_probes.py --model gemma-2-9b --activations ../activations/
    python train_probes.py --model all --activations ../activations/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm


class LinearProbe(nn.Module):
    """Linear probe for empathy type classification."""

    def __init__(self, d_model: int, n_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.linear(x)


def load_activations(filepath: Path) -> Dict:
    """Load activation data for a model."""
    with open(filepath, 'r') as f:
        return json.load(f)


def prepare_probe_data(activations_data: Dict, layer: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for probe training.

    Returns:
        X: activations array [n_samples, d_model]
        y: labels array [n_samples] (0=cognitive, 1=affective, 2=instrumental)
        label_names: list of label names
    """
    items = activations_data['items']

    # Filter to empathy triplets only
    empathy_items = [item for item in items if item['dataset'] == 'empathy_triplets']

    X_list = []
    y_list = []

    label_map = {
        'cognitive': 0,
        'affective': 1,
        'instrumental': 2
    }

    for item in empathy_items:
        if str(layer) in item['layers']:
            activation = np.array(item['layers'][str(layer)])
            response_type = item['response_type']

            if response_type in label_map:
                X_list.append(activation)
                y_list.append(label_map[response_type])

    X = np.array(X_list)
    y = np.array(y_list)

    label_names = ['cognitive', 'affective', 'instrumental']

    return X, y, label_names


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    d_model: int,
    n_classes: int = 3,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cuda'
) -> LinearProbe:
    """Train a linear probe for empathy classification."""

    probe = LinearProbe(d_model, n_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    best_val_acc = 0
    best_probe_state = None

    # Training loop
    for epoch in range(n_epochs):
        probe.train()

        # Forward pass
        logits = probe(X_train_t)
        loss = criterion(logits, y_train_t)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val_t)
            val_loss = criterion(val_logits, y_val_t)

            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_probe_state = probe.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}, "
                  f"Val Acc={val_acc:.4f}")

    # Restore best probe
    probe.load_state_dict(best_probe_state)

    return probe


def evaluate_probe(
    probe: LinearProbe,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
    device: str = 'cuda'
) -> Dict:
    """Evaluate probe performance."""

    probe.eval()

    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        logits = probe(X_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    # Accuracy
    accuracy = accuracy_score(y_test, preds)

    # Multi-class AUROC (one-vs-rest)
    auroc_scores = {}
    for i, label in enumerate(label_names):
        y_binary = (y_test == i).astype(int)
        auroc = roc_auc_score(y_binary, probs[:, i])
        auroc_scores[label] = float(auroc)

    # Macro-average AUROC
    macro_auroc = np.mean(list(auroc_scores.values()))

    # Classification report
    report = classification_report(y_test, preds, target_names=label_names, output_dict=True)

    results = {
        'accuracy': float(accuracy),
        'auroc_per_class': auroc_scores,
        'macro_auroc': float(macro_auroc),
        'classification_report': report
    }

    return results


def extract_probe_vectors(probe: LinearProbe) -> Dict[str, np.ndarray]:
    """
    Extract direction vectors from probe for convergence analysis.

    Returns dict mapping label to direction vector.
    """
    weight_matrix = probe.linear.weight.detach().cpu().numpy()  # [n_classes, d_model]

    vectors = {
        'cognitive': weight_matrix[0],
        'affective': weight_matrix[1],
        'instrumental': weight_matrix[2]
    }

    return vectors


def compute_separation_metrics(probe_vectors: Dict[str, np.ndarray]) -> Dict:
    """
    Compute separation metrics between probe directions.

    Key metric: cosine(Cognitive, Affective) < 0.5 indicates separation.
    """
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    cognitive = probe_vectors['cognitive']
    affective = probe_vectors['affective']
    instrumental = probe_vectors['instrumental']

    metrics = {
        'cosine_cognitive_affective': float(cosine_similarity(cognitive, affective)),
        'cosine_cognitive_instrumental': float(cosine_similarity(cognitive, instrumental)),
        'cosine_affective_instrumental': float(cosine_similarity(affective, instrumental))
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train probes for empathy classification')
    parser.add_argument('--model', type=str, default='all',
                        help='Model to train probes for (or "all")')
    parser.add_argument('--activations', type=str, default='../activations',
                        help='Directory containing activation files')
    parser.add_argument('--output', type=str, default='../results/experiment_b',
                        help='Output directory for probe results')
    parser.add_argument('--layer', type=int, default=None,
                        help='Specific layer to train (default: middle layer)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Setup paths
    activations_dir = Path(args.activations)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find activation files
    if args.model == 'all':
        activation_files = list(activations_dir.glob('*_activations.json'))
    else:
        activation_files = [activations_dir / f'{args.model}_activations.json']

    print(f"Found {len(activation_files)} activation file(s)")

    # Train probes for each model
    all_results = {}

    for act_file in activation_files:
        model_name = act_file.stem.replace('_activations', '')

        print(f"\n{'='*60}")
        print(f"Training probe for: {model_name}")
        print(f"{'='*60}")

        # Load activations
        activations_data = load_activations(act_file)
        n_layers = activations_data['n_layers']
        d_model = activations_data['d_model']

        print(f"Model: {n_layers} layers, d_model={d_model}")

        # Determine layer
        if args.layer is not None:
            target_layer = args.layer
        else:
            target_layer = n_layers // 2

        print(f"Training probe on layer {target_layer}")

        # Prepare data
        X, y, label_names = prepare_probe_data(activations_data, target_layer)
        print(f"Prepared {X.shape[0]} samples")

        # Train/val/test split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
        )

        print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        # Train probe
        print("\nTraining probe...")
        probe = train_probe(
            X_train, y_train, X_val, y_val,
            d_model, n_classes=3,
            n_epochs=args.epochs,
            device=args.device
        )

        # Evaluate
        print("\nEvaluating probe...")
        results = evaluate_probe(probe, X_test, y_test, label_names, args.device)

        print(f"\nResults:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Macro AUROC: {results['macro_auroc']:.4f}")
        print(f"  AUROC per class:")
        for label, auroc in results['auroc_per_class'].items():
            print(f"    {label}: {auroc:.4f}")

        # Extract probe vectors
        probe_vectors = extract_probe_vectors(probe)

        # Compute separation metrics
        separation = compute_separation_metrics(probe_vectors)

        print(f"\nSeparation metrics:")
        for metric, value in separation.items():
            print(f"  {metric}: {value:.4f}")

        # Save results
        model_results = {
            'model_name': model_name,
            'layer': target_layer,
            'd_model': d_model,
            'performance': results,
            'separation_metrics': separation,
            'probe_vectors': {k: v.tolist() for k, v in probe_vectors.items()}
        }

        all_results[model_name] = model_results

        # Save individual model results
        output_file = output_dir / f'{model_name}_layer{target_layer}_probe.json'
        with open(output_file, 'w') as f:
            json.dump(model_results, f, indent=2)

        print(f"\n✓ Saved results to: {output_file}")

    # Save summary
    summary_file = output_dir / 'probe_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Probe training complete!")
    print(f"Summary: {summary_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
