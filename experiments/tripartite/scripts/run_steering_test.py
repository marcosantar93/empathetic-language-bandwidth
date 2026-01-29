#!/usr/bin/env python3
"""
Steering Test: Apply empathy direction vectors and observe output changes.
Tests whether Cognitive-Affective direction actually shifts response style.
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_model(model_name, device='cuda'):
    """Load model with TransformerLens."""
    from transformer_lens import HookedTransformer

    model_map = {
        'mistral-7b': 'mistralai/Mistral-7B-v0.1',
        'llama-3-8b': 'meta-llama/Meta-Llama-3-8B',
        'gemma-7b': 'google/gemma-7b',
    }

    hf_name = model_map.get(model_name, model_name)
    model = HookedTransformer.from_pretrained(
        hf_name,
        device=device,
        torch_dtype=torch.float16
    )
    return model

def extract_direction(model, prompts_a, prompts_b, layer, device='cuda'):
    """Extract direction vector as mean(A) - mean(B)."""

    def get_mean_activation(prompts):
        acts = []
        for prompt in tqdm(prompts, desc="Extracting"):
            tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=f'blocks.{layer}.hook_resid_post')
            act = cache[f'blocks.{layer}.hook_resid_post'][0, -1, :].cpu().numpy()
            acts.append(act)
        return np.mean(acts, axis=0)

    mean_a = get_mean_activation(prompts_a)
    mean_b = get_mean_activation(prompts_b)

    direction = mean_a - mean_b
    direction = direction / np.linalg.norm(direction)  # Normalize
    return direction

def generate_with_steering(model, prompt, direction, layer, alpha, max_tokens=100, device='cuda'):
    """Generate text with steering applied at specified layer."""

    direction_tensor = torch.tensor(direction, dtype=torch.float16, device=device)

    def steering_hook(activation, hook):
        # Add direction to last token position
        activation[:, -1, :] += alpha * direction_tensor
        return activation

    tokens = model.to_tokens(prompt, prepend_bos=True).to(device)

    # Generate with hook
    with model.hooks(fwd_hooks=[(f'blocks.{layer}.hook_resid_post', steering_hook)]):
        output = model.generate(
            tokens,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    return model.to_string(output[0])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mistral-7b')
    parser.add_argument('--data-dir', default='experiments/tripartite/data')
    parser.add_argument('--output', default='experiments/tripartite/results/steering_test.json')
    parser.add_argument('--layer', type=int, default=16)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    data_dir = Path(args.data_dir)

    # Load data
    with open(data_dir / 'triplets_filtered.json') as f:
        data = json.load(f)['triplets']

    # Prepare prompts for direction extraction
    cognitive_prompts = []
    affective_prompts = []

    for item in data[:20]:  # Use 20 for direction extraction
        scenario = item['scenario']
        if 'cognitive' in item:
            cognitive_prompts.append(f"{scenario}\n\nResponse: {item['cognitive']}")
        if 'affective' in item:
            affective_prompts.append(f"{scenario}\n\nResponse: {item['affective']}")

    print(f"\nLoading model...")
    model = load_model(args.model, device)

    # Extract Cognitive - Affective direction
    print(f"\nExtracting Cognitive-Affective direction at layer {args.layer}...")
    cog_aff_direction = extract_direction(
        model, cognitive_prompts, affective_prompts, args.layer, device
    )

    # Test scenarios (use different ones than training)
    test_scenarios = [item['scenario'] for item in data[60:65]]

    # Steering coefficients to test
    alphas = [-3.0, -1.5, 0.0, 1.5, 3.0]

    results = {
        'model': args.model,
        'layer': args.layer,
        'direction': 'cognitive_minus_affective',
        'alphas': alphas,
        'generations': []
    }

    print(f"\n{'='*60}")
    print("STEERING TEST")
    print("Direction: Cognitive - Affective")
    print("Positive alpha → more cognitive (perspective-taking)")
    print("Negative alpha → more affective (emotional warmth)")
    print(f"{'='*60}\n")

    for scenario in test_scenarios:
        prompt = f"{scenario}\n\nResponse:"
        print(f"\nScenario: {scenario[:100]}...")
        print("-" * 40)

        scenario_results = {'scenario': scenario, 'outputs': {}}

        for alpha in alphas:
            output = generate_with_steering(
                model, prompt, cog_aff_direction, args.layer, alpha,
                max_tokens=80, device=device
            )
            # Extract just the response part
            response = output.split("Response:")[-1].strip()
            scenario_results['outputs'][str(alpha)] = response

            direction_label = "COGNITIVE" if alpha > 0 else "AFFECTIVE" if alpha < 0 else "NEUTRAL"
            print(f"\nα={alpha:+.1f} ({direction_label}):")
            print(f"  {response[:200]}...")

        results['generations'].append(scenario_results)

    # Save results
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved: {args.output}")

if __name__ == '__main__':
    main()
