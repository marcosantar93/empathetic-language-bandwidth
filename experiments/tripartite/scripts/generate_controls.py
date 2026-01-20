#!/usr/bin/env python3
"""
Generate control datasets for tripartite empathy decomposition research.

Two control conditions:
1. non-empathy: Emotional but non-empathetic scenarios (60 items)
   - Excitement, aesthetic appreciation, intellectual curiosity, etc.
   - Validates that we're measuring empathy, not just emotional language

2. valence-stripped: Same scenarios as main dataset, neutral responses (90 items)
   - Factual, informative, but emotionally flat
   - Validates that we're measuring empathetic content, not just helpfulness

Usage:
    python generate_controls.py non-empathy --output ../data/controls_non_empathy.json --count 60
    python generate_controls.py valence-stripped --output ../data/controls_valence_stripped.json --input ../data/triplets_filtered.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from anthropic import Anthropic
from tqdm import tqdm


NON_EMPATHY_SCENARIO_PROMPT = """Generate a realistic scenario that evokes emotional responses but does NOT require empathy. The scenario should:

1. Evoke emotions like: excitement, aesthetic appreciation, intellectual curiosity, anticipation, satisfaction, awe, or righteous anger at abstract issues
2. NOT involve interpersonal distress or someone needing emotional support
3. NOT involve understanding another person's emotional state
4. Be detailed and engaging (2-3 sentences)
5. Be diverse across domains: nature, art, science, politics, personal achievement, discovery, etc.

Examples of non-empathetic emotional scenarios:
- Discovering an elegant mathematical proof
- Witnessing a stunning sunset over the ocean
- Learning about a breakthrough in renewable energy
- Achieving a personal fitness goal
- Finding a rare book in a used bookstore

Return ONLY the scenario text, nothing else.
"""


NON_EMPATHY_RESPONSE_PROMPT = """Given this non-empathetic emotional scenario:

{scenario}

Generate 3 emotional responses that DO NOT involve empathy:

**Response 1 (Enthusiasm):**
- Express excitement or appreciation
- Focus on the subject matter, not interpersonal connection
- 2-3 sentences

**Response 2 (Analytical):**
- Intellectual engagement with the topic
- Curious, thoughtful, but not emotionally warm toward a person
- 2-3 sentences

**Response 3 (Reflective):**
- Personal reflection or philosophical musing
- Introspective but not empathetic
- 2-3 sentences

Return your response in this EXACT JSON format (no markdown, no extra text):
{{
  "response_1": "Your enthusiastic response here",
  "response_2": "Your analytical response here",
  "response_3": "Your reflective response here"
}}
"""


VALENCE_STRIPPED_PROMPT = """Given this empathy-requiring scenario:

{scenario}

Generate 3 neutral, factual responses that strip away empathetic and emotional content:

**Response 1 (Factual):**
- State objective facts about the situation
- No emotional warmth, no perspective-taking
- Clinical, detached tone
- 2-3 sentences

**Response 2 (Informational):**
- Provide general information related to the situation
- Educational but impersonal
- 2-3 sentences

**Response 3 (Procedural):**
- Describe standard procedures or typical outcomes
- Bureaucratic, process-focused tone
- 2-3 sentences

Return your response in this EXACT JSON format (no markdown, no extra text):
{{
  "factual": "Your factual response here",
  "informational": "Your informational response here",
  "procedural": "Your procedural response here"
}}
"""


def load_triplets(filepath: Path) -> List[Dict]:
    """Load triplets from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data.get('triplets', [])


def load_existing_controls(filepath: Path) -> List[Dict]:
    """Load existing control items if resuming."""
    if filepath.exists():
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data.get('items', [])
    return []


def save_controls(items: List[Dict], filepath: Path, control_type: str, metadata: Optional[Dict] = None):
    """Save control items to JSON with metadata."""
    output = {
        'metadata': metadata or {
            'total_items': len(items),
            'control_type': control_type,
            'version': '1.0',
            'experiment': 'tripartite_decomposition'
        },
        'items': items
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


def generate_non_empathy_scenario(client: Anthropic, model: str = "claude-3-5-haiku-20241022") -> str:
    """Generate a single non-empathetic emotional scenario."""
    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=1.0,
        messages=[{"role": "user", "content": NON_EMPATHY_SCENARIO_PROMPT}]
    )
    return response.content[0].text.strip()


def generate_non_empathy_responses(client: Anthropic, scenario: str,
                                    model: str = "claude-3-5-haiku-20241022") -> Dict[str, str]:
    """Generate non-empathetic emotional responses."""
    prompt = NON_EMPATHY_RESPONSE_PROMPT.format(scenario=scenario)

    response = client.messages.create(
        model=model,
        max_tokens=800,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text.strip()

    # Handle markdown wrapping
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    if response_text.startswith('```'):
        response_text = response_text[3:]
    if response_text.endswith('```'):
        response_text = response_text[:-3]

    response_text = response_text.strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"\nError parsing response: {e}")
        print(f"Response text: {response_text[:200]}...")
        raise


def generate_valence_stripped_responses(client: Anthropic, scenario: str,
                                        model: str = "claude-3-5-haiku-20241022") -> Dict[str, str]:
    """Generate valence-stripped responses for empathy scenario."""
    prompt = VALENCE_STRIPPED_PROMPT.format(scenario=scenario)

    response = client.messages.create(
        model=model,
        max_tokens=800,
        temperature=0.5,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text.strip()

    # Handle markdown wrapping
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    if response_text.startswith('```'):
        response_text = response_text[3:]
    if response_text.endswith('```'):
        response_text = response_text[:-3]

    response_text = response_text.strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"\nError parsing response: {e}")
        print(f"Response text: {response_text[:200]}...")
        raise


def generate_non_empathy_item(client: Anthropic, item_id: int,
                               model: str = "claude-3-5-haiku-20241022",
                               max_retries: int = 3) -> Optional[Dict]:
    """Generate a single non-empathy control item."""
    for attempt in range(max_retries):
        try:
            scenario = generate_non_empathy_scenario(client, model)
            responses = generate_non_empathy_responses(client, scenario, model)

            return {
                'id': item_id,
                'scenario': scenario,
                'response_1': responses['response_1'],
                'response_2': responses['response_2'],
                'response_3': responses['response_3'],
                'control_type': 'non_empathy',
                'attempt': attempt + 1
            }

        except Exception as e:
            print(f"\nError on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return None

    return None


def generate_valence_stripped_item(client: Anthropic, triplet: Dict, item_id: int,
                                    model: str = "claude-3-5-haiku-20241022",
                                    max_retries: int = 3) -> Optional[Dict]:
    """Generate valence-stripped responses for an empathy scenario."""
    for attempt in range(max_retries):
        try:
            responses = generate_valence_stripped_responses(client, triplet['scenario'], model)

            return {
                'id': item_id,
                'triplet_id': triplet['id'],
                'scenario': triplet['scenario'],
                'factual': responses['factual'],
                'informational': responses['informational'],
                'procedural': responses['procedural'],
                'control_type': 'valence_stripped',
                'attempt': attempt + 1
            }

        except Exception as e:
            print(f"\nError on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return None

    return None


def main():
    parser = argparse.ArgumentParser(description='Generate control datasets')
    parser.add_argument('mode', choices=['non-empathy', 'valence-stripped'],
                        help='Control condition to generate')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--input', type=str, default=None,
                        help='Input triplets file (required for valence-stripped mode)')
    parser.add_argument('--count', type=int, default=60,
                        help='Number of items to generate (non-empathy mode only)')
    parser.add_argument('--model', type=str, default='claude-3-5-haiku-20241022',
                        help='Claude model to use')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output file')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Anthropic API key (or use ANTHROPIC_API_KEY env var)')

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'valence-stripped' and not args.input:
        print("Error: --input is required for valence-stripped mode")
        sys.exit(1)

    # Initialize client
    api_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set. Use --api-key or set environment variable.")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    # Load existing items if resuming
    output_path = Path(args.output)
    if args.resume:
        existing_items = load_existing_controls(output_path)
        print(f"Resuming from {len(existing_items)} existing items")
    else:
        existing_items = []

    if args.mode == 'non-empathy':
        start_id = len(existing_items)
        remaining = args.count - start_id

        if remaining <= 0:
            print(f"Already have {start_id} items (target: {args.count}). Nothing to do.")
            return

        print(f"Generating {remaining} non-empathy items (IDs {start_id} to {args.count - 1})")
        print(f"Model: {args.model}")
        print(f"Output: {output_path}")
        print()

        items = existing_items.copy()

        for i in tqdm(range(start_id, args.count), desc="Generating non-empathy items"):
            item = generate_non_empathy_item(client, i, args.model)

            if item:
                items.append(item)

                # Save periodically
                if (i + 1) % 10 == 0:
                    save_controls(items, output_path, 'non_empathy', {
                        'total_items': len(items),
                        'target_count': args.count,
                        'control_type': 'non_empathy',
                        'model': args.model,
                        'version': '1.0',
                        'status': 'in_progress'
                    })
            else:
                print(f"\nFailed to generate item {i} after max retries")

        # Final save
        save_controls(items, output_path, 'non_empathy', {
            'total_items': len(items),
            'target_count': args.count,
            'control_type': 'non_empathy',
            'model': args.model,
            'version': '1.0',
            'status': 'complete'
        })

        print(f"\n{'='*60}")
        print(f"Non-empathy generation complete!")
        print(f"Total items: {len(items)}")
        print(f"Output: {output_path}")
        print(f"{'='*60}")

    else:  # valence-stripped mode
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)

        triplets = load_triplets(input_path)
        print(f"Loaded {len(triplets)} triplets from {input_path}")

        start_id = len(existing_items)
        remaining = len(triplets) - start_id

        if remaining <= 0:
            print(f"Already processed all {len(triplets)} triplets. Nothing to do.")
            return

        print(f"Generating {remaining} valence-stripped items (IDs {start_id} to {len(triplets) - 1})")
        print(f"Model: {args.model}")
        print(f"Output: {output_path}")
        print()

        items = existing_items.copy()

        for i in tqdm(range(start_id, len(triplets)), desc="Generating valence-stripped items"):
            item = generate_valence_stripped_item(client, triplets[i], i, args.model)

            if item:
                items.append(item)

                # Save periodically
                if (i + 1) % 10 == 0:
                    save_controls(items, output_path, 'valence_stripped', {
                        'total_items': len(items),
                        'source_triplets': str(input_path),
                        'control_type': 'valence_stripped',
                        'model': args.model,
                        'version': '1.0',
                        'status': 'in_progress'
                    })
            else:
                print(f"\nFailed to generate item {i} after max retries")

        # Final save
        save_controls(items, output_path, 'valence_stripped', {
            'total_items': len(items),
            'source_triplets': str(input_path),
            'control_type': 'valence_stripped',
            'model': args.model,
            'version': '1.0',
            'status': 'complete'
        })

        print(f"\n{'='*60}")
        print(f"Valence-stripped generation complete!")
        print(f"Total items: {len(items)}")
        print(f"Output: {output_path}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
