#!/usr/bin/env python3
"""
Extract activations from LLMs for empathy triplet datasets.

Uses TransformerLens to extract residual stream activations at each layer
for all scenarios and responses across all datasets.

Models:
- Gemma2-9B
- Llama-3.1-8B
- Qwen2.5-7B
- Mistral-7B
- DeepSeek-R1-7B

Usage:
    python extract_activations.py --model gemma-2-9b --output ../activations/
    python extract_activations.py --model all --output ../activations/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer


# Model name mappings
MODEL_NAMES = {
    'gemma-2-9b': 'google/gemma-2-9b',
    'llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B',
    'qwen2.5-7b': 'Qwen/Qwen2.5-7B',
    'mistral-7b': 'mistralai/Mistral-7B-v0.1',
    'deepseek-r1-7b': 'deepseek-ai/deepseek-coder-7b-base-v1.5'  # Placeholder - adjust when R1 available
}


def load_datasets(data_dir: Path) -> Dict[str, List[Dict]]:
    """Load all tripartite datasets."""
    datasets = {}

    # Empathy triplets
    with open(data_dir / 'triplets_filtered.json', 'r') as f:
        data = json.load(f)
        datasets['empathy_triplets'] = data['triplets']

    # Non-empathy controls
    with open(data_dir / 'controls_non_empathy.json', 'r') as f:
        data = json.load(f)
        datasets['non_empathy'] = data['items']

    # Valence-stripped controls
    with open(data_dir / 'controls_valence_stripped.json', 'r') as f:
        data = json.load(f)
        datasets['valence_stripped'] = data['items']

    return datasets


def prepare_prompts(datasets: Dict[str, List[Dict]]) -> List[Tuple[str, str, str, int, str]]:
    """
    Prepare all prompts for activation extraction.

    Returns list of (prompt, dataset_name, response_type, item_id, scenario) tuples.
    """
    prompts = []

    # Empathy triplets
    for item in datasets['empathy_triplets']:
        scenario = item['scenario']
        for response_type in ['cognitive', 'affective', 'instrumental']:
            response = item[response_type]
            prompt = f"Scenario: {scenario}\n\nResponse: {response}"
            prompts.append((prompt, 'empathy_triplets', response_type, item['id'], scenario))

    # Non-empathy controls
    for item in datasets['non_empathy']:
        scenario = item['scenario']
        for i, response_type in enumerate(['response_1', 'response_2', 'response_3'], 1):
            response = item[response_type]
            prompt = f"Scenario: {scenario}\n\nResponse: {response}"
            prompts.append((prompt, 'non_empathy', f'response_{i}', item['id'], scenario))

    # Valence-stripped controls
    for item in datasets['valence_stripped']:
        scenario = item['scenario']
        for response_type in ['factual', 'informational', 'procedural']:
            response = item[response_type]
            prompt = f"Scenario: {scenario}\n\nResponse: {response}"
            prompts.append((prompt, 'valence_stripped', response_type, item['id'], scenario))

    return prompts


def extract_activations_for_model(
    model_name: str,
    prompts: List[Tuple[str, str, str, int, str]],
    device: str = 'cuda',
    batch_size: int = 8
) -> Dict:
    """
    Extract residual stream activations for all prompts.

    Returns dict with activations and metadata.
    """
    print(f"\nLoading model: {MODEL_NAMES[model_name]}")

    # Load model with TransformerLens
    model = HookedTransformer.from_pretrained(
        MODEL_NAMES[model_name],
        device=device,
        dtype=torch.float16 if device == 'cuda' else torch.float32
    )

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    print(f"Model loaded: {n_layers} layers, d_model={d_model}")

    # Storage for activations
    activations_data = {
        'model': model_name,
        'n_layers': n_layers,
        'd_model': d_model,
        'items': []
    }

    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Extracting {model_name}"):
        batch = prompts[i:i + batch_size]
        batch_prompts = [p[0] for p in batch]

        # Tokenize
        tokens = model.to_tokens(batch_prompts, prepend_bos=True)

        # Extract activations at each layer
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Extract residual stream activations at each layer
        for j, (prompt, dataset, response_type, item_id, scenario) in enumerate(batch):
            item_activations = {
                'dataset': dataset,
                'response_type': response_type,
                'item_id': item_id,
                'scenario': scenario[:100],  # Truncate for storage
                'prompt_length': tokens.shape[1],
                'layers': {}
            }

            # Get activations at each layer (averaged across sequence)
            for layer in range(n_layers):
                # Residual stream at this layer
                resid = cache[f'blocks.{layer}.hook_resid_post'][j]  # [seq_len, d_model]

                # Average across sequence (excluding padding)
                avg_activation = resid.mean(dim=0).cpu().numpy()  # [d_model]

                item_activations['layers'][layer] = avg_activation.tolist()

            activations_data['items'].append(item_activations)

    # Clean up
    del model
    torch.cuda.empty_cache()

    return activations_data


def main():
    parser = argparse.ArgumentParser(description='Extract activations from LLMs')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all'] + list(MODEL_NAMES.keys()),
                        help='Model to extract activations from')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Directory containing datasets')
    parser.add_argument('--output', type=str, default='../activations',
                        help='Output directory for activations')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    datasets = load_datasets(data_dir)

    total_items = sum(len(d) for d in datasets.values())
    print(f"Loaded {len(datasets)} datasets with {total_items} total items")

    # Prepare prompts
    print("Preparing prompts...")
    prompts = prepare_prompts(datasets)
    print(f"Prepared {len(prompts)} prompts for activation extraction")

    # Determine models to process
    if args.model == 'all':
        models_to_process = list(MODEL_NAMES.keys())
    else:
        models_to_process = [args.model]

    print(f"\nProcessing {len(models_to_process)} model(s)")

    # Extract activations for each model
    for model_name in models_to_process:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")

        try:
            activations_data = extract_activations_for_model(
                model_name,
                prompts,
                device=args.device,
                batch_size=args.batch_size
            )

            # Save activations
            output_file = output_dir / f'{model_name}_activations.json'
            print(f"Saving activations to: {output_file}")

            with open(output_file, 'w') as f:
                json.dump(activations_data, f)

            print(f"✓ Saved {len(activations_data['items'])} activation sets")

        except Exception as e:
            print(f"✗ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("Activation extraction complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
