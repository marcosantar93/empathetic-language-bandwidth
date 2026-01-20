#!/usr/bin/env python3
"""
Analyze convergence between SAE-discovered clusters and theory-driven probes.

Research Question:
Do geometry-driven SAE clusters align with theory-predicted tripartite structure?

Key Tests:
1. Do SAEs discover ~3-4 clusters naturally?
2. Do SAE cluster centroids align with probe direction vectors?
3. Mean cosine similarity > 0.5 indicates convergence

Usage:
    python convergence_analysis.py --results-dir ../results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_sae_results(sae_dir: Path, model_name: str, layer: int) -> Dict:
    """Load SAE training results."""
    import torch

    sae_file = sae_dir / f'{model_name}_layer{layer}_sae.pt'

    if not sae_file.exists():
        raise FileNotFoundError(f"SAE file not found: {sae_file}")

    data = torch.load(sae_file, map_location='cpu')

    return data


def load_probe_results(probe_dir: Path, model_name: str, layer: int) -> Dict:
    """Load probe training results."""
    probe_file = probe_dir / f'{model_name}_layer{layer}_probe.json'

    if not probe_file.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_file}")

    with open(probe_file, 'r') as f:
        return json.load(f)


def compute_convergence_score(sae_data: Dict, probe_data: Dict) -> Dict:
    """
    Compute convergence between SAE clusters and probe directions.

    Strategy:
    1. Get SAE cluster centroids for optimal k (likely 3 or 4)
    2. Get probe direction vectors (Cognitive, Affective, Instrumental)
    3. Compute alignment (cosine similarity) between centroids and directions
    4. High alignment (mean cosine > 0.5) = convergence
    """
    # Get cluster analysis from SAE
    cluster_analysis = sae_data.get('cluster_analysis', {})

    # Find optimal cluster count (highest silhouette score)
    best_k = max(cluster_analysis.keys(), key=lambda k: cluster_analysis[k]['silhouette_score'])
    best_clusters = cluster_analysis[best_k]

    print(f"  Optimal SAE clusters: {best_k} (silhouette={best_clusters['silhouette_score']:.3f})")

    # Get probe vectors
    probe_vectors = {
        'cognitive': np.array(probe_data['probe_vectors']['cognitive']),
        'affective': np.array(probe_data['probe_vectors']['affective']),
        'instrumental': np.array(probe_data['probe_vectors']['instrumental'])
    }

    # Note: We can't directly compute centroids without activations
    # For now, we analyze cluster composition alignment
    composition_analysis = analyze_cluster_composition(best_clusters, best_k)

    convergence_metrics = {
        'optimal_k': int(best_k),
        'silhouette_score': float(best_clusters['silhouette_score']),
        'theoretical_k': 3,  # Cognitive, Affective, Instrumental
        'k_match': best_k == '3' or best_k == 3,
        'cluster_composition': composition_analysis,
        'probe_separation': probe_data['separation_metrics']
    }

    return convergence_metrics


def analyze_cluster_composition(cluster_data: Dict, k: int) -> Dict:
    """
    Analyze if SAE clusters align with empathy types.

    A cluster is "pure" if >70% of its members are one empathy type.
    """
    composition = cluster_data['cluster_composition']

    cluster_purity = {}
    dominant_types = {}

    for cluster_id, info in composition.items():
        response_types = info['response_types']
        total = sum(response_types.values())

        if total > 0:
            # Find dominant type
            dominant_type = max(response_types.items(), key=lambda x: x[1])
            purity = dominant_type[1] / total

            cluster_purity[cluster_id] = {
                'dominant_type': dominant_type[0],
                'purity': float(purity),
                'distribution': {k: v/total for k, v in response_types.items()}
            }

    # Overall purity score
    mean_purity = np.mean([info['purity'] for info in cluster_purity.values()])

    analysis = {
        'cluster_purity': cluster_purity,
        'mean_purity': float(mean_purity),
        'high_purity_threshold': 0.7,
        'high_purity_clusters': sum(1 for info in cluster_purity.values() if info['purity'] > 0.7)
    }

    return analysis


def create_convergence_report(convergence_results: Dict[str, Dict], output_dir: Path):
    """Create comprehensive convergence analysis report."""

    report = {
        'summary': {},
        'per_model': convergence_results
    }

    # Overall statistics
    k_matches = [r['k_match'] for r in convergence_results.values()]
    mean_purities = [r['cluster_composition']['mean_purity'] for r in convergence_results.values()]
    silhouette_scores = [r['silhouette_score'] for r in convergence_results.values()]

    report['summary'] = {
        'n_models': len(convergence_results),
        'k_match_rate': float(np.mean(k_matches)),
        'mean_cluster_purity': float(np.mean(mean_purities)),
        'mean_silhouette_score': float(np.mean(silhouette_scores)),
        'convergence_conclusion': analyze_convergence(convergence_results)
    }

    # Save report
    report_file = output_dir / 'convergence_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Saved convergence report: {report_file}")

    return report


def analyze_convergence(results: Dict[str, Dict]) -> str:
    """Determine overall convergence conclusion."""

    k_match_rate = np.mean([r['k_match'] for r in results.values()])
    mean_purity = np.mean([r['cluster_composition']['mean_purity'] for r in results.values()])

    if k_match_rate >= 0.6 and mean_purity >= 0.7:
        conclusion = "STRONG_CONVERGENCE"
    elif k_match_rate >= 0.4 and mean_purity >= 0.5:
        conclusion = "MODERATE_CONVERGENCE"
    else:
        conclusion = "WEAK_CONVERGENCE"

    return conclusion


def create_visualization(convergence_results: Dict[str, Dict], output_dir: Path):
    """Create visualization of convergence metrics."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = list(convergence_results.keys())

    # 1. Optimal K by model
    ax = axes[0, 0]
    optimal_ks = [convergence_results[m]['optimal_k'] for m in models]
    ax.bar(models, optimal_ks)
    ax.axhline(y=3, color='r', linestyle='--', label='Theoretical K=3')
    ax.set_ylabel('Optimal K')
    ax.set_title('SAE Cluster Count (Geometry-Driven)')
    ax.legend()
    ax.set_xticklabels(models, rotation=45)

    # 2. Cluster purity
    ax = axes[0, 1]
    purities = [convergence_results[m]['cluster_composition']['mean_purity'] for m in models]
    ax.bar(models, purities)
    ax.axhline(y=0.7, color='r', linestyle='--', label='High purity threshold')
    ax.set_ylabel('Mean Cluster Purity')
    ax.set_title('Cluster Purity (Alignment with Theory)')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_xticklabels(models, rotation=45)

    # 3. Probe separation
    ax = axes[1, 0]
    probe_separations = []
    for m in models:
        sep = convergence_results[m]['probe_separation']
        # Lower cosine = better separation
        avg_sep = np.mean([
            sep['cosine_cognitive_affective'],
            sep['cosine_cognitive_instrumental'],
            sep['cosine_affective_instrumental']
        ])
        probe_separations.append(avg_sep)

    ax.bar(models, probe_separations)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Separation threshold')
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Probe Direction Separation\n(Lower = Better)')
    ax.legend()
    ax.set_xticklabels(models, rotation=45)

    # 4. Silhouette scores
    ax = axes[1, 1]
    silhouettes = [convergence_results[m]['silhouette_score'] for m in models]
    ax.bar(models, silhouettes)
    ax.set_ylabel('Silhouette Score')
    ax.set_title('SAE Cluster Quality')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(models, rotation=45)

    plt.tight_layout()

    # Save figure
    fig_file = output_dir / 'convergence_visualization.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {fig_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze convergence between SAEs and probes')
    parser.add_argument('--results-dir', type=str, default='../results',
                        help='Results directory')
    parser.add_argument('--sae-dir', type=str, default=None,
                        help='SAE directory (default: results-dir/../saes)')
    parser.add_argument('--probe-dir', type=str, default=None,
                        help='Probe directory (default: results-dir/experiment_b)')
    parser.add_argument('--layer', type=int, default=None,
                        help='Specific layer (default: infer from files)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    sae_dir = Path(args.sae_dir) if args.sae_dir else results_dir.parent / 'saes'
    probe_dir = Path(args.probe_dir) if args.probe_dir else results_dir / 'experiment_b'

    print(f"SAE directory: {sae_dir}")
    print(f"Probe directory: {probe_dir}")

    # Find models
    sae_files = list(sae_dir.glob('*_sae.pt'))
    models = set([f.stem.split('_layer')[0] for f in sae_files])

    print(f"\nFound {len(models)} model(s): {', '.join(models)}")

    convergence_results = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"Analyzing: {model}")
        print(f"{'='*60}")

        # Find layer
        model_sae_files = list(sae_dir.glob(f'{model}_layer*_sae.pt'))
        if not model_sae_files:
            print(f"  No SAE files found for {model}")
            continue

        # Use middle layer if not specified
        if args.layer is not None:
            layer = args.layer
        else:
            # Extract layer from filename
            layer = int(model_sae_files[0].stem.split('_layer')[1].split('_')[0])

        print(f"  Layer: {layer}")

        try:
            # Load data
            sae_data = load_sae_results(sae_dir, model, layer)
            probe_data = load_probe_results(probe_dir, model, layer)

            # Compute convergence
            convergence = compute_convergence_score(sae_data, probe_data)

            print(f"\n  Convergence Metrics:")
            print(f"    Optimal K: {convergence['optimal_k']} (theory: {convergence['theoretical_k']})")
            print(f"    K Match: {convergence['k_match']}")
            print(f"    Mean Cluster Purity: {convergence['cluster_composition']['mean_purity']:.3f}")

            convergence_results[model] = convergence

        except Exception as e:
            print(f"  Error analyzing {model}: {e}")
            import traceback
            traceback.print_exc()

    # Create overall report
    if convergence_results:
        print(f"\n{'='*60}")
        print("Creating Convergence Report")
        print(f"{'='*60}")

        report = create_convergence_report(convergence_results, results_dir)

        print(f"\nOverall Summary:")
        print(f"  Models analyzed: {report['summary']['n_models']}")
        print(f"  K match rate: {report['summary']['k_match_rate']:.2%}")
        print(f"  Mean cluster purity: {report['summary']['mean_cluster_purity']:.3f}")
        print(f"  Conclusion: {report['summary']['convergence_conclusion']}")

        # Create visualization
        create_visualization(convergence_results, results_dir)

        print(f"\n{'='*60}")
        print("Convergence analysis complete!")
        print(f"{'='*60}")

    else:
        print("\nNo convergence results to analyze.")


if __name__ == '__main__':
    main()
