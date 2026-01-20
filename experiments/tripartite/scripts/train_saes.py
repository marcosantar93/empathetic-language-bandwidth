#!/usr/bin/env python3
"""
Train Sparse Autoencoders (SAEs) on empathy activation data.

Experiment A: Geometry-driven discovery
- Let SAEs discover natural clusters in empathy activation space
- Test if discovered clusters align with theoretical tripartite structure

Usage:
    python train_saes.py --model gemma-2-9b --activations ../activations/
    python train_saes.py --model all --activations ../activations/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for learning empathy activation structure."""

    def __init__(self, d_model: int, n_features: int = 512, sparsity_coef: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.sparsity_coef = sparsity_coef

        # Encoder
        self.encoder = nn.Linear(d_model, n_features)

        # Decoder
        self.decoder = nn.Linear(n_features, d_model)

    def forward(self, x):
        # Encode
        features = torch.relu(self.encoder(x))

        # Decode
        reconstruction = self.decoder(features)

        return reconstruction, features

    def loss(self, x, reconstruction, features):
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(reconstruction, x)

        # Sparsity loss (L1 on features)
        sparsity_loss = torch.mean(torch.abs(features))

        # Combined loss
        total_loss = recon_loss + self.sparsity_coef * sparsity_loss

        return total_loss, recon_loss, sparsity_loss


def load_activations(filepath: Path) -> Dict:
    """Load activation data for a model."""
    with open(filepath, 'r') as f:
        return json.load(f)


def prepare_training_data(activations_data: Dict, layer: int) -> tuple:
    """
    Prepare training data for SAE at specific layer.

    Returns:
        activations_array: np.array of shape [n_samples, d_model]
        metadata: list of dicts with dataset/response_type info
    """
    items = activations_data['items']

    activations_list = []
    metadata_list = []

    for item in items:
        if str(layer) in item['layers']:
            activation = np.array(item['layers'][str(layer)])
            activations_list.append(activation)

            metadata_list.append({
                'dataset': item['dataset'],
                'response_type': item['response_type'],
                'item_id': item['item_id']
            })

    activations_array = np.array(activations_list)

    return activations_array, metadata_list


def train_sae(
    activations: np.ndarray,
    d_model: int,
    n_features: int = 512,
    sparsity_coef: float = 0.01,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = 'cuda'
) -> SparseAutoencoder:
    """Train a Sparse Autoencoder on activation data."""

    # Initialize SAE
    sae = SparseAutoencoder(d_model, n_features, sparsity_coef).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=lr)

    # Prepare data
    n_samples = activations.shape[0]
    activations_tensor = torch.FloatTensor(activations).to(device)

    # Training loop
    sae.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_recon = 0
        epoch_sparsity = 0

        # Shuffle data
        indices = torch.randperm(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = activations_tensor[batch_indices]

            # Forward pass
            reconstruction, features = sae(batch)

            # Compute loss
            loss, recon_loss, sparsity_loss = sae.loss(batch, reconstruction, features)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_sparsity += sparsity_loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / (n_samples // batch_size)
            avg_recon = epoch_recon / (n_samples // batch_size)
            avg_sparsity = epoch_sparsity / (n_samples // batch_size)
            print(f"  Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, "
                  f"Recon={avg_recon:.4f}, Sparsity={avg_sparsity:.4f}")

    return sae


def analyze_sae_clusters(
    sae: SparseAutoencoder,
    activations: np.ndarray,
    metadata: List[Dict],
    device: str = 'cuda'
) -> Dict:
    """
    Analyze SAE feature space to discover natural clusters.

    Tests:
    - How many natural clusters exist?
    - Do they align with theoretical structure (Cognitive/Affective/Instrumental)?
    """
    sae.eval()

    with torch.no_grad():
        activations_tensor = torch.FloatTensor(activations).to(device)
        _, features = sae(activations_tensor)
        features_np = features.cpu().numpy()

    # Test different cluster counts
    cluster_analysis = {}

    for n_clusters in [2, 3, 4, 5, 6]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_np)

        # Silhouette score (higher is better)
        silhouette = silhouette_score(features_np, cluster_labels)

        # Analyze cluster composition
        cluster_composition = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_items = [metadata[j] for j in range(len(metadata)) if cluster_mask[j]]

            # Count by response type
            response_types = {}
            for item in cluster_items:
                rt = item['response_type']
                response_types[rt] = response_types.get(rt, 0) + 1

            cluster_composition[i] = {
                'size': int(cluster_mask.sum()),
                'response_types': response_types
            }

        cluster_analysis[n_clusters] = {
            'silhouette_score': float(silhouette),
            'cluster_composition': cluster_composition,
            'cluster_labels': cluster_labels.tolist()
        }

    return cluster_analysis


def main():
    parser = argparse.ArgumentParser(description='Train SAEs for empathy geometry discovery')
    parser.add_argument('--model', type=str, default='all',
                        help='Model to train SAEs for (or "all")')
    parser.add_argument('--activations', type=str, default='../activations',
                        help='Directory containing activation files')
    parser.add_argument('--output', type=str, default='../saes',
                        help='Output directory for trained SAEs')
    parser.add_argument('--layer', type=int, default=None,
                        help='Specific layer to train (default: middle layer)')
    parser.add_argument('--n-features', type=int, default=512,
                        help='Number of SAE features')
    parser.add_argument('--sparsity', type=float, default=0.01,
                        help='Sparsity coefficient')
    parser.add_argument('--epochs', type=int, default=100,
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

    # Train SAE for each model
    for act_file in activation_files:
        model_name = act_file.stem.replace('_activations', '')

        print(f"\n{'='*60}")
        print(f"Training SAE for: {model_name}")
        print(f"{'='*60}")

        # Load activations
        activations_data = load_activations(act_file)
        n_layers = activations_data['n_layers']
        d_model = activations_data['d_model']

        print(f"Model: {n_layers} layers, d_model={d_model}")

        # Determine layer to train on
        if args.layer is not None:
            target_layer = args.layer
        else:
            target_layer = n_layers // 2  # Middle layer

        print(f"Training SAE on layer {target_layer}")

        # Prepare training data
        activations, metadata = prepare_training_data(activations_data, target_layer)
        print(f"Prepared {activations.shape[0]} activation samples")

        # Train SAE
        print("\nTraining SAE...")
        sae = train_sae(
            activations,
            d_model,
            n_features=args.n_features,
            sparsity_coef=args.sparsity,
            n_epochs=args.epochs,
            device=args.device
        )

        # Analyze clusters
        print("\nAnalyzing SAE clusters...")
        cluster_analysis = analyze_sae_clusters(sae, activations, metadata, args.device)

        # Print results
        print("\nCluster Analysis:")
        for n_clusters, analysis in cluster_analysis.items():
            print(f"\n  {n_clusters} clusters: Silhouette={analysis['silhouette_score']:.3f}")
            for cluster_id, comp in analysis['cluster_composition'].items():
                print(f"    Cluster {cluster_id} (n={comp['size']}): {comp['response_types']}")

        # Save SAE and analysis
        output_file = output_dir / f'{model_name}_layer{target_layer}_sae.pt'
        torch.save({
            'model_name': model_name,
            'layer': target_layer,
            'd_model': d_model,
            'n_features': args.n_features,
            'sparsity_coef': args.sparsity,
            'state_dict': sae.state_dict(),
            'cluster_analysis': cluster_analysis
        }, output_file)

        print(f"\n✓ Saved SAE to: {output_file}")

    print(f"\n{'='*60}")
    print("SAE training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
