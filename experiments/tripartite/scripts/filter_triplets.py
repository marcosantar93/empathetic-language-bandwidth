#!/usr/bin/env python3
"""
Filter and curate triplets for tripartite empathy research.

Reviews all triplets with multi-criteria evaluation:
- Clarity of tripartite distinction (Cognitive/Affective/Instrumental)
- Scenario quality and realism
- Response quality and naturalness
- Domain diversity

Usage:
    python filter_triplets.py --input ../data/triplets_raw.json --output ../data/triplets_filtered.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from anthropic import Anthropic
from tqdm import tqdm


COUNCIL_REVIEW_PROMPT = """You are part of a research council reviewing empathy triplets for a rigorous neuroscience study.

Evaluate this triplet for research quality:

**Scenario:** {scenario}

**Cognitive Response:** {cognitive}

**Affective Response:** {affective}

**Instrumental Response:** {instrumental}

**Evaluation Criteria:**

1. **Tripartite Distinction (Most Important)**
   - Are the three responses clearly distinct in their empathetic dimension?
   - Cognitive: Focus on understanding mental states, perspective-taking (no warmth)
   - Affective: Focus on emotional resonance, warmth, feeling-with (minimal analysis)
   - Instrumental: Focus on practical solutions (minimal emotional connection)
   - Score: CLEAR (excellent separation) | MODERATE (some overlap) | WEAK (poor separation)

2. **Scenario Quality**
   - Realistic and relatable?
   - Appropriate emotional depth (not too extreme, not too trivial)?
   - Clear enough for responses?
   - Score: EXCELLENT | GOOD | ACCEPTABLE | POOR

3. **Response Quality**
   - Natural and human-sounding?
   - Appropriate length and detail?
   - All three responses high-quality?
   - Score: EXCELLENT | GOOD | ACCEPTABLE | POOR

4. **Research Value**
   - Will this help measure empathetic activation geometry?
   - Clear enough signal for neural differentiation?
   - Score: HIGH | MEDIUM | LOW

**Overall Decision:**
- ACCEPT: Include in filtered dataset (high research value)
- ACCEPT_WITH_EDIT: Good but needs minor improvements (specify what)
- REJECT: Does not meet research standards (explain why)

Return your evaluation in this EXACT JSON format:
{{
  "tripartite_distinction": "CLEAR|MODERATE|WEAK",
  "scenario_quality": "EXCELLENT|GOOD|ACCEPTABLE|POOR",
  "response_quality": "EXCELLENT|GOOD|ACCEPTABLE|POOR",
  "research_value": "HIGH|MEDIUM|LOW",
  "decision": "ACCEPT|ACCEPT_WITH_EDIT|REJECT",
  "reasoning": "Brief explanation of your decision (1-2 sentences)",
  "edit_suggestions": "What to improve (if ACCEPT_WITH_EDIT)" or null
}}
"""


def load_triplets(filepath: Path) -> List[Dict]:
    """Load triplets from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data.get('triplets', [])


def save_filtered_triplets(triplets: List[Dict], filepath: Path, metadata: Dict):
    """Save filtered triplets to JSON."""
    output = {
        'metadata': metadata,
        'triplets': triplets
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


def council_review(client: Anthropic, triplet: Dict, model: str = "claude-3-5-haiku-20241022") -> Dict:
    """Get council evaluation of a triplet."""
    prompt = COUNCIL_REVIEW_PROMPT.format(
        scenario=triplet['scenario'],
        cognitive=triplet['cognitive'],
        affective=triplet['affective'],
        instrumental=triplet['instrumental']
    )

    response = client.messages.create(
        model=model,
        max_tokens=600,
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
        # Try to extract just the JSON object
        try:
            # Find the first { and last } to extract just the JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                json_only = response_text[start:end]
                return json.loads(json_only)
        except:
            pass

        print(f"\nError parsing council review: {e}")
        print(f"Response: {response_text[:200]}...")
        return {
            "decision": "REJECT",
            "reasoning": "Failed to parse council review",
            "tripartite_distinction": "WEAK",
            "scenario_quality": "POOR",
            "response_quality": "POOR",
            "research_value": "LOW",
            "edit_suggestions": None
        }


def main():
    parser = argparse.ArgumentParser(description='Filter triplets for research quality')
    parser.add_argument('--input', type=str, default='../data/triplets_raw.json',
                        help='Input triplets file')
    parser.add_argument('--output', type=str, default='../data/triplets_filtered.json',
                        help='Output filtered triplets file')
    parser.add_argument('--model', type=str, default='claude-3-5-haiku-20241022',
                        help='Claude model for council review')
    parser.add_argument('--auto-accept-validated', action='store_true',
                        help='Auto-accept triplets that passed initial validation')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Anthropic API key (or use ANTHROPIC_API_KEY env var)')

    args = parser.parse_args()

    # Initialize client
    api_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set. Use --api-key or set environment variable.")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    # Load triplets
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    triplets = load_triplets(input_path)
    print(f"Loaded {len(triplets)} triplets from {input_path}")

    # Categorize triplets
    if args.auto_accept_validated:
        validated = [t for t in triplets if t.get('validation', {}).get('valid', False)]
        needs_review = [t for t in triplets if not t.get('validation', {}).get('valid', False)]
        print(f"\nAuto-accepting {len(validated)} validated triplets")
        print(f"Council reviewing {len(needs_review)} flagged triplets")
    else:
        validated = []
        needs_review = triplets
        print(f"\nCouncil reviewing all {len(needs_review)} triplets")

    # Council review
    accepted = []
    accepted_with_edits = []
    rejected = []

    for triplet in tqdm(needs_review, desc="Council review"):
        review = council_review(client, triplet, args.model)

        triplet['council_review'] = review

        if review['decision'] == 'ACCEPT':
            accepted.append(triplet)
        elif review['decision'] == 'ACCEPT_WITH_EDIT':
            accepted_with_edits.append(triplet)
        else:
            rejected.append(triplet)

    # Combine results
    filtered_triplets = validated + accepted + accepted_with_edits

    # Re-number triplets
    for i, triplet in enumerate(filtered_triplets):
        triplet['original_id'] = triplet['id']
        triplet['id'] = i

    # Statistics
    print(f"\n{'='*60}")
    print("FILTERING COMPLETE")
    print(f"{'='*60}")
    if args.auto_accept_validated:
        print(f"Auto-accepted (validated): {len(validated)}")
    print(f"Council accepted: {len(accepted)}")
    print(f"Accepted with edit notes: {len(accepted_with_edits)}")
    print(f"Rejected: {len(rejected)}")
    print(f"{'='*60}")
    print(f"Total filtered triplets: {len(filtered_triplets)}")
    print(f"{'='*60}")

    # Quality breakdown
    if accepted or accepted_with_edits:
        reviewed = accepted + accepted_with_edits
        clear_distinction = sum(1 for t in reviewed if t.get('council_review', {}).get('tripartite_distinction') == 'CLEAR')
        high_value = sum(1 for t in reviewed if t.get('council_review', {}).get('research_value') == 'HIGH')

        print(f"\nQuality Metrics (Council Reviewed):")
        print(f"  Clear tripartite distinction: {clear_distinction}/{len(reviewed)}")
        print(f"  High research value: {high_value}/{len(reviewed)}")

    # Show rejection reasons
    if rejected:
        print(f"\nRejection Reasons:")
        for t in rejected[:5]:  # Show first 5
            review = t.get('council_review', {})
            print(f"  - Triplet {t['id']}: {review.get('reasoning', 'Unknown')}")
        if len(rejected) > 5:
            print(f"  ... and {len(rejected) - 5} more")

    # Show edit suggestions
    if accepted_with_edits:
        print(f"\nEdit Suggestions (first 5):")
        for t in accepted_with_edits[:5]:
            review = t.get('council_review', {})
            if review.get('edit_suggestions'):
                print(f"  - Triplet {t['id']}: {review['edit_suggestions'][:80]}...")
        if len(accepted_with_edits) > 5:
            print(f"  ... and {len(accepted_with_edits) - 5} more")

    # Save filtered dataset
    output_path = Path(args.output)
    metadata = {
        'total_triplets': len(filtered_triplets),
        'source_file': str(input_path),
        'filtering_model': args.model,
        'version': '1.0',
        'experiment': 'tripartite_decomposition',
        'auto_accepted': len(validated),
        'council_accepted': len(accepted),
        'accepted_with_edits': len(accepted_with_edits),
        'rejected_count': len(rejected),
        'quality_metrics': {
            'clear_distinction': sum(1 for t in filtered_triplets if t.get('council_review', {}).get('tripartite_distinction') == 'CLEAR'),
            'high_research_value': sum(1 for t in filtered_triplets if t.get('council_review', {}).get('research_value') == 'HIGH')
        }
    }

    save_filtered_triplets(filtered_triplets, output_path, metadata)
    print(f"\nFiltered triplets saved to: {output_path}")


if __name__ == '__main__':
    main()
