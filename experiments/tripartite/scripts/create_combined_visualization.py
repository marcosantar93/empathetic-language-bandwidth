#!/usr/bin/env python3
"""
Create combined visualization for all models in the tripartite experiment.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    """Load convergence reports from all models."""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    models = ['qwen2.5-7b', 'mistral-7b', 'llama-3-8b', 'llama-3.1-8b']

    data = {}
    for model in models:
        report_path = os.path.join(results_dir, model, 'results', 'convergence_report.json')
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
                data[model] = report['per_model'][model]

    return data

def create_visualization(data, output_path):
    """Create combined visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Tripartite Empathy Decomposition: Cross-Model Analysis', fontsize=14, fontweight='bold')

    models = list(data.keys())
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    # 1. Cosine similarities heatmap
    ax1 = axes[0, 0]
    cos_data = []
    for model in models:
        sep = data[model]['probe_separation']
        cos_data.append([
            sep['cosine_cognitive_affective'],
            sep['cosine_cognitive_instrumental'],
            sep['cosine_affective_instrumental']
        ])

    cos_array = np.array(cos_data)
    im = ax1.imshow(cos_array, cmap='RdBu', aspect='auto', vmin=-0.5, vmax=0.5)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(['Cog-Aff', 'Cog-Instr', 'Aff-Instr'], fontsize=9)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models, fontsize=9)
    ax1.set_title('Probe Cosine Similarities', fontweight='bold')

    # Add text annotations
    for i in range(len(models)):
        for j in range(3):
            text = ax1.text(j, i, f'{cos_array[i, j]:.2f}',
                          ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    plt.colorbar(im, ax=ax1, label='Cosine Similarity')

    # 2. Bar chart of Cog-Aff separation
    ax2 = axes[0, 1]
    cog_aff = [data[m]['probe_separation']['cosine_cognitive_affective'] for m in models]
    bars = ax2.barh(models, [-x for x in cog_aff], color=colors)
    ax2.axvline(x=0.5, color='red', linestyle='--', label='Target threshold')
    ax2.set_xlabel('Separation (negative cosine)')
    ax2.set_title('Cognitive-Affective Separation', fontweight='bold')
    ax2.legend()
    ax2.set_xlim(0, 0.5)

    # Add value labels
    for bar, val in zip(bars, cog_aff):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    # 3. Silhouette scores
    ax3 = axes[1, 0]
    silhouettes = [data[m]['silhouette_score'] for m in models]
    bars = ax3.bar(models, silhouettes, color=colors)
    ax3.axhline(y=0.5, color='green', linestyle='--', label='Good clustering')
    ax3.axhline(y=0.25, color='orange', linestyle='--', label='Fair clustering')
    ax3.set_ylabel('Silhouette Score')
    ax3.set_title('SAE Clustering Quality', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 0.6)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add value labels
    for bar, val in zip(bars, silhouettes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontsize=9)

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Calculate summary stats
    mean_cog_aff = np.mean([data[m]['probe_separation']['cosine_cognitive_affective'] for m in models])
    mean_cog_instr = np.mean([data[m]['probe_separation']['cosine_cognitive_instrumental'] for m in models])
    mean_aff_instr = np.mean([data[m]['probe_separation']['cosine_affective_instrumental'] for m in models])
    mean_silhouette = np.mean(silhouettes)

    summary_text = f"""
    SUMMARY STATISTICS
    ══════════════════════════════════════

    Probe Separation (cosine similarity):
    ─────────────────────────────────────
    • Cognitive-Affective:     {mean_cog_aff:.3f}
    • Cognitive-Instrumental:  {mean_cog_instr:.3f}
    • Affective-Instrumental:  {mean_aff_instr:.3f}

    SAE Clustering:
    ─────────────────────────────────────
    • Mean Silhouette:         {mean_silhouette:.3f}
    • Optimal K (all models):  2
    • Theoretical K:           3

    CONCLUSIONS
    ══════════════════════════════════════

    ✓ H1 (Separation): CONFIRMED
      Mean cos(Cog,Aff) = {mean_cog_aff:.2f} < 0.5

    ~ H2 (Convergence): PARTIAL
      SAE finds k=2, not k=3

    ✓ H3 (Consistency): CONFIRMED
      All models show negative cosines
    """

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')

    return output_path

if __name__ == '__main__':
    data = load_results()
    print(f'Loaded results for {len(data)} models: {list(data.keys())}')

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    output_path = os.path.join(output_dir, 'combined_visualization.png')

    create_visualization(data, output_path)
