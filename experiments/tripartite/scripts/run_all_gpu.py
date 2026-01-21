#!/usr/bin/env python3
"""
Run all GPU-based experiments for tripartite empathy decomposition.

Orchestrates:
1. Activation extraction from all models
2. SAE training (Experiment A: geometry-driven)
3. Probe training (Experiment B: theory-driven)
4. Convergence analysis

This is the main entry point for Docker container execution on EC2.

Usage:
    python run_all_gpu.py --models all
    python run_all_gpu.py --models gemma-2-9b,llama-3.1-8b
    python run_all_gpu.py --skip-extraction  # If activations already extracted
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


MODELS = [
    'gemma-2-9b',
    'llama-3.1-8b',
    'qwen2.5-7b',
    'mistral-7b',
    'deepseek-r1-7b'
]


def run_command(cmd: list, description: str, check: bool = True):
    """Run a command with logging."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\n✗ {description} failed with exit code {result.returncode}")
        if check:
            sys.exit(1)
    else:
        print(f"\n✓ {description} completed successfully")

    return result.returncode


def extract_activations(models: list, data_dir: Path, output_dir: Path, device: str, batch_size: int):
    """Run activation extraction for all models."""
    activations_dir = output_dir / 'activations'
    activations_dir.mkdir(parents=True, exist_ok=True)

    # Get script directory
    script_dir = Path(__file__).parent

    for model in models:
        cmd = [
            'python', str(script_dir / 'extract_activations.py'),
            '--model', model,
            '--data-dir', str(data_dir),
            '--output', str(activations_dir),
            '--batch-size', str(batch_size),
            '--device', device
        ]

        run_command(cmd, f"Extracting activations: {model}")


def train_saes(models: list, activations_dir: Path, output_dir: Path, device: str, epochs: int):
    """Run SAE training (Experiment A)."""
    saes_dir = output_dir / 'saes'
    saes_dir.mkdir(parents=True, exist_ok=True)

    # Get script directory
    script_dir = Path(__file__).parent

    for model in models:
        cmd = [
            'python', str(script_dir / 'train_saes.py'),
            '--model', model,
            '--activations', str(activations_dir),
            '--output', str(saes_dir),
            '--epochs', str(epochs),
            '--device', device
        ]

        run_command(cmd, f"Training SAE: {model}")


def train_probes(models: list, activations_dir: Path, output_dir: Path, device: str, epochs: int):
    """Run probe training (Experiment B)."""
    probes_dir = output_dir / 'results' / 'experiment_b'
    probes_dir.mkdir(parents=True, exist_ok=True)

    # Get script directory
    script_dir = Path(__file__).parent

    for model in models:
        cmd = [
            'python', str(script_dir / 'train_probes.py'),
            '--model', model,
            '--activations', str(activations_dir),
            '--output', str(probes_dir),
            '--epochs', str(epochs),
            '--device', device
        ]

        run_command(cmd, f"Training probe: {model}")


def run_convergence_analysis(results_dir: Path):
    """Run convergence analysis."""
    # Get script directory
    script_dir = Path(__file__).parent

    cmd = [
        'python', str(script_dir / 'convergence_analysis.py'),
        '--results-dir', str(results_dir)
    ]

    run_command(cmd, "Convergence analysis")


def create_experiment_summary(output_dir: Path, models: list, start_time: datetime, end_time: datetime):
    """Create summary of experiment run."""
    duration = (end_time - start_time).total_seconds()

    summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration,
        'duration_hours': duration / 3600,
        'models_processed': models,
        'n_models': len(models),
        'experiment': 'tripartite_decomposition',
        'version': '1.0'
    }

    # Check for result files
    activations_dir = output_dir / 'activations'
    saes_dir = output_dir / 'saes'
    probes_dir = output_dir / 'results' / 'experiment_b'

    summary['outputs'] = {
        'activations': len(list(activations_dir.glob('*.json'))) if activations_dir.exists() else 0,
        'saes': len(list(saes_dir.glob('*.pt'))) if saes_dir.exists() else 0,
        'probes': len(list(probes_dir.glob('*.json'))) if probes_dir.exists() else 0,
        'convergence_report': (output_dir / 'results' / 'convergence_report.json').exists()
    }

    summary_file = output_dir / 'results' / 'experiment_summary.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Saved experiment summary: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Run all tripartite empathy experiments')
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated model names or "all"')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Directory containing datasets')
    parser.add_argument('--output-dir', type=str, default='..',
                        help='Output directory (parent of activations, saes, results)')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip activation extraction (use existing)')
    parser.add_argument('--skip-saes', action='store_true',
                        help='Skip SAE training')
    parser.add_argument('--skip-probes', action='store_true',
                        help='Skip probe training')
    parser.add_argument('--skip-convergence', action='store_true',
                        help='Skip convergence analysis')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for activation extraction')
    parser.add_argument('--sae-epochs', type=int, default=100,
                        help='SAE training epochs')
    parser.add_argument('--probe-epochs', type=int, default=50,
                        help='Probe training epochs')

    args = parser.parse_args()

    # Parse models
    if args.models == 'all':
        models = MODELS
    else:
        models = [m.strip() for m in args.models.split(',')]

    # Validate models
    invalid_models = [m for m in models if m not in MODELS]
    if invalid_models:
        print(f"Error: Invalid model(s): {', '.join(invalid_models)}")
        print(f"Valid models: {', '.join(MODELS)}")
        sys.exit(1)

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    activations_dir = output_dir / 'activations'
    results_dir = output_dir / 'results'

    # Start experiment
    start_time = datetime.now()

    print("="*70)
    print("TRIPARTITE EMPATHY DECOMPOSITION - GPU EXPERIMENTS")
    print("="*70)
    print(f"Start time: {start_time}")
    print(f"Models: {', '.join(models)}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print("="*70)

    try:
        # Step 1: Extract activations
        if not args.skip_extraction:
            print("\n\n" + "="*70)
            print("STEP 1: ACTIVATION EXTRACTION")
            print("="*70)
            extract_activations(models, data_dir, output_dir, args.device, args.batch_size)
        else:
            print("\nSkipping activation extraction (using existing)")

        # Step 2: Train SAEs (Experiment A)
        if not args.skip_saes:
            print("\n\n" + "="*70)
            print("STEP 2: EXPERIMENT A - SAE TRAINING (Geometry-Driven)")
            print("="*70)
            train_saes(models, activations_dir, output_dir, args.device, args.sae_epochs)
        else:
            print("\nSkipping SAE training")

        # Step 3: Train Probes (Experiment B)
        if not args.skip_probes:
            print("\n\n" + "="*70)
            print("STEP 3: EXPERIMENT B - PROBE TRAINING (Theory-Driven)")
            print("="*70)
            train_probes(models, activations_dir, output_dir, args.device, args.probe_epochs)
        else:
            print("\nSkipping probe training")

        # Step 4: Convergence Analysis
        if not args.skip_convergence:
            print("\n\n" + "="*70)
            print("STEP 4: CONVERGENCE ANALYSIS")
            print("="*70)
            run_convergence_analysis(results_dir)
        else:
            print("\nSkipping convergence analysis")

        # Complete
        end_time = datetime.now()

        print("\n\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETE!")
        print("="*70)

        summary = create_experiment_summary(output_dir, models, start_time, end_time)

        print(f"\nDuration: {summary['duration_hours']:.2f} hours")
        print(f"Outputs:")
        print(f"  Activations: {summary['outputs']['activations']}")
        print(f"  SAEs: {summary['outputs']['saes']}")
        print(f"  Probes: {summary['outputs']['probes']}")
        print(f"  Convergence report: {summary['outputs']['convergence_report']}")

        print("\n" + "="*70)
        print("SUCCESS - Ready for analysis and reporting")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
