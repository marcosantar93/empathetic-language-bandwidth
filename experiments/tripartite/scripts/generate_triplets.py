#!/usr/bin/env python3
"""
Generate triplet dataset for tripartite empathy decomposition research.

Creates 90 scenarios, each with 3 response types:
- Cognitive empathy: Perspective-taking, understanding mental states
- Affective empathy: Emotional resonance, warmth, feeling with others
- Instrumental: Helpful but detached, problem-solving focus

Usage:
    python generate_triplets.py --output ../data/triplets_raw.json --count 90
    python generate_triplets.py --output ../data/triplets_raw.json --count 90 --resume
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from anthropic import Anthropic
from tqdm import tqdm


SCENARIO_GENERATION_PROMPT = """Generate a realistic interpersonal scenario that would elicit empathetic responses. The scenario should:

1. Involve emotional or psychological distress (not just practical problems)
2. Be detailed enough to understand the person's situation (2-3 sentences)
3. Be diverse across domains: work, relationships, family, health, identity, loss, etc.
4. Avoid extreme trauma (keep it relatable but meaningful)
5. Include enough context for perspective-taking

Return ONLY the scenario text, nothing else. Example format:
"A colleague mentions that their partner recently asked for space after a 5-year relationship. They seem confused about what went wrong and whether they should reach out or wait. They're having trouble concentrating at work."
"""


RESPONSE_GENERATION_PROMPT = """Given this scenario:

{scenario}

Generate 3 distinct empathetic responses that differ in their empathetic dimension:

**Cognitive Empathy Response:**
- Focus on understanding and perspective-taking
- Acknowledge the person's mental state and reasoning
- Describe what they might be thinking or experiencing
- Analytical, insightful, but emotionally neutral tone
- 2-3 sentences

**Affective Empathy Response:**
- Focus on emotional resonance and warmth
- Express feeling with the person, not just for them
- Use warm, emotionally attuned language
- Convey care and emotional connection
- 2-3 sentences

**Instrumental Response:**
- Focus on practical help and problem-solving
- Offer actionable advice or resources
- Helpful but emotionally detached
- Professional, efficient tone
- 2-3 sentences

Return your response in this EXACT JSON format (no markdown, no extra text):
{{
  "cognitive": "Your cognitive empathy response here",
  "affective": "Your affective empathy response here",
  "instrumental": "Your instrumental response here"
}}
"""


RUBRIC_VALIDATION_PROMPT = """Evaluate if these responses meet the tripartite empathy criteria:

Scenario: {scenario}

Cognitive: {cognitive}
Affective: {affective}
Instrumental: {instrumental}

Check:
1. Are the three responses clearly distinct in their empathetic dimension?
2. Does cognitive focus on mental states without emotional warmth?
3. Does affective express emotional resonance and warmth?
4. Does instrumental prioritize solutions over emotional connection?
5. Are all responses high-quality and natural-sounding?

Respond with JSON:
{{
  "valid": true/false,
  "issues": ["list", "of", "problems"] or [],
  "suggestions": "how to improve" or ""
}}
"""


def load_existing_triplets(filepath: Path) -> List[Dict]:
    """Load existing triplets if resuming."""
    if filepath.exists():
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data.get('triplets', [])
    return []


def save_triplets(triplets: List[Dict], filepath: Path, metadata: Optional[Dict] = None):
    """Save triplets to JSON with metadata."""
    output = {
        'metadata': metadata or {
            'total_triplets': len(triplets),
            'version': '1.0',
            'experiment': 'tripartite_decomposition'
        },
        'triplets': triplets
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


def generate_scenario(client: Anthropic, model: str = "claude-3-5-haiku-20241022") -> str:
    """Generate a single empathy-eliciting scenario."""
    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=1.0,
        messages=[{"role": "user", "content": SCENARIO_GENERATION_PROMPT}]
    )
    return response.content[0].text.strip()


def generate_responses(client: Anthropic, scenario: str, model: str = "claude-3-5-haiku-20241022") -> Dict[str, str]:
    """Generate cognitive, affective, and instrumental responses for a scenario."""
    prompt = RESPONSE_GENERATION_PROMPT.format(scenario=scenario)

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text.strip()

    # Handle potential markdown wrapping
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    if response_text.startswith('```'):
        response_text = response_text[3:]
    if response_text.endswith('```'):
        response_text = response_text[:-3]

    response_text = response_text.strip()

    try:
        responses = json.loads(response_text)
        return {
            'cognitive': responses['cognitive'],
            'affective': responses['affective'],
            'instrumental': responses['instrumental']
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"\nError parsing response: {e}")
        print(f"Response text: {response_text[:200]}...")
        raise


def validate_triplet(client: Anthropic, scenario: str, responses: Dict[str, str],
                     model: str = "claude-3-5-haiku-20241022") -> Dict:
    """Validate that responses meet tripartite criteria."""
    prompt = RUBRIC_VALIDATION_PROMPT.format(
        scenario=scenario,
        cognitive=responses['cognitive'],
        affective=responses['affective'],
        instrumental=responses['instrumental']
    )

    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0.3,
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
        print(f"\nError parsing validation: {e}")
        return {"valid": False, "issues": ["JSON parse error"], "suggestions": ""}


def generate_triplet(client: Anthropic, triplet_id: int, model: str = "claude-3-5-haiku-20241022",
                     max_retries: int = 3) -> Optional[Dict]:
    """Generate a single triplet with validation and retries."""
    for attempt in range(max_retries):
        try:
            # Generate scenario
            scenario = generate_scenario(client, model)

            # Generate responses
            responses = generate_responses(client, scenario, model)

            # Validate
            validation = validate_triplet(client, scenario, responses, model)

            triplet = {
                'id': triplet_id,
                'scenario': scenario,
                'cognitive': responses['cognitive'],
                'affective': responses['affective'],
                'instrumental': responses['instrumental'],
                'validation': validation,
                'attempt': attempt + 1
            }

            if validation.get('valid', False):
                return triplet
            else:
                # Include invalid triplets for manual review
                triplet['needs_review'] = True
                return triplet

        except Exception as e:
            print(f"\nError on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return None

    return None


def main():
    parser = argparse.ArgumentParser(description='Generate empathy triplet dataset')
    parser.add_argument('--output', type=str, default='../data/triplets_raw.json',
                        help='Output JSON file path')
    parser.add_argument('--count', type=int, default=90,
                        help='Number of triplets to generate')
    parser.add_argument('--model', type=str, default='claude-3-5-haiku-20241022',
                        help='Claude model to use')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output file')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Anthropic API key (or use ANTHROPIC_API_KEY env var)')

    args = parser.parse_args()

    # Initialize client
    api_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set. Use --api-key or set environment variable.")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    # Load existing triplets if resuming
    output_path = Path(args.output)
    if args.resume:
        existing_triplets = load_existing_triplets(output_path)
        print(f"Resuming from {len(existing_triplets)} existing triplets")
    else:
        existing_triplets = []

    start_id = len(existing_triplets)
    remaining = args.count - start_id

    if remaining <= 0:
        print(f"Already have {start_id} triplets (target: {args.count}). Nothing to do.")
        return

    print(f"Generating {remaining} triplets (IDs {start_id} to {args.count - 1})")
    print(f"Model: {args.model}")
    print(f"Output: {output_path}")
    print()

    triplets = existing_triplets.copy()

    # Generate triplets with progress bar
    for i in tqdm(range(start_id, args.count), desc="Generating triplets"):
        triplet = generate_triplet(client, i, args.model)

        if triplet:
            triplets.append(triplet)

            # Save periodically (every 10 triplets)
            if (i + 1) % 10 == 0:
                save_triplets(triplets, output_path, {
                    'total_triplets': len(triplets),
                    'target_count': args.count,
                    'model': args.model,
                    'version': '1.0',
                    'experiment': 'tripartite_decomposition',
                    'status': 'in_progress'
                })
        else:
            print(f"\nFailed to generate triplet {i} after max retries")

    # Final save
    save_triplets(triplets, output_path, {
        'total_triplets': len(triplets),
        'target_count': args.count,
        'model': args.model,
        'version': '1.0',
        'experiment': 'tripartite_decomposition',
        'status': 'complete',
        'valid_count': sum(1 for t in triplets if t.get('validation', {}).get('valid', False)),
        'needs_review_count': sum(1 for t in triplets if t.get('needs_review', False))
    })

    # Summary
    valid_count = sum(1 for t in triplets if t.get('validation', {}).get('valid', False))
    review_count = sum(1 for t in triplets if t.get('needs_review', False))

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Total triplets: {len(triplets)}")
    print(f"Validated: {valid_count}")
    print(f"Needs review: {review_count}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    if review_count > 0:
        print(f"\nNext step: Manually review {review_count} triplets and create triplets_filtered.json")


if __name__ == '__main__':
    main()
