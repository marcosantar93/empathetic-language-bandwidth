#!/usr/bin/env python3
"""
RunPod Jupyter Launcher - Most Reliable Approach

Based on extensive RunPod debugging:
- Start pods with Jupyter Lab working
- Provide clear instructions for running experiments
- User can monitor progress in real-time
- No complex automation that fails silently

Usage:
    python runpod_jupyter.py
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional

import requests

RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
HF_TOKEN = os.environ.get('HF_TOKEN')

if not RUNPOD_API_KEY:
    print("Error: RUNPOD_API_KEY not set")
    sys.exit(1)

RUNPOD_API_BASE = "https://api.runpod.io/graphql"
GPU_TYPES = ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090"]
MODELS = ['llama-3.1-8b', 'qwen2.5-7b', 'mistral-7b', 'llama-3-8b']


def runpod_query(query: str, variables: Optional[Dict] = None) -> Dict:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {RUNPOD_API_KEY}'
    }
    payload = {'query': query}
    if variables:
        payload['variables'] = variables
    response = requests.post(RUNPOD_API_BASE, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    if 'errors' in result:
        raise Exception(f"GraphQL errors: {result['errors']}")
    return result['data']


def create_jupyter_pod(name: str, model: str) -> Optional[Dict]:
    """Create pod with Jupyter Lab - simple and reliable."""

    query = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            name
            machine {
                gpuDisplayName
            }
        }
    }
    """

    for gpu_type in GPU_TYPES:
        try:
            variables = {
                'input': {
                    'name': name,
                    'imageName': 'pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime',
                    'gpuTypeId': gpu_type,
                    'cloudType': 'COMMUNITY',
                    'gpuCount': 1,
                    'volumeInGb': 50,
                    'containerDiskInGb': 50,
                    'dockerArgs': '',  # Empty = use RunPod defaults
                    'ports': '8888/http',
                    'volumeMountPath': '/workspace',
                    'env': [
                        {'key': 'JUPYTER_PASSWORD', 'value': 'runpod'},
                        {'key': 'HF_TOKEN', 'value': HF_TOKEN},
                        {'key': 'MODEL', 'value': model},
                    ]
                }
            }

            print(f"   Trying {gpu_type}...")
            result = runpod_query(query, variables)
            pod = result['podFindAndDeployOnDemand']
            print(f"   ✅ Success: {pod['id']} ({pod['machine']['gpuDisplayName']})")

            return {
                'id': pod['id'],
                'name': pod['name'],
                'gpu': pod['machine']['gpuDisplayName'],
                'model': model
            }

        except Exception as e:
            if "no longer any instances available" in str(e).lower():
                print(f"   ⚠️ {gpu_type} not available")
                continue
            print(f"   ❌ Error: {e}")
            continue

    return None


def print_instructions(pods: List[Dict]):
    """Print clear instructions for running experiments."""

    commands = f"""
# Install dependencies
pip install -q huggingface_hub[cli]
huggingface-cli login --token {HF_TOKEN}

# Clone repository
cd /workspace
git clone https://github.com/marcosantar93/empathetic-language-bandwidth.git
cd empathetic-language-bandwidth
git checkout main

# Install requirements
pip install -q --no-cache-dir -r requirements-gpu.txt

# Run experiment for your model
# For MODEL in pods, run:
python experiments/tripartite/scripts/run_all_gpu.py --models $MODEL --batch-size 2
"""

    print("\n" + "="*70)
    print("HOW TO RUN EXPERIMENTS")
    print("="*70)
    print("\n1. Access Jupyter Lab for each pod:")
    print("   https://www.runpod.io/console/pods")
    print("   Password: runpod")
    print("\n2. Open a terminal in Jupyter Lab")
    print("\n3. Run these commands:")
    print(commands)
    print("\n4. Monitor progress in the terminal")
    print("\n5. Results will be in:")
    print("   /workspace/empathetic-language-bandwidth/experiments/tripartite/results/")


def main():
    print("\n" + "="*70)
    print("RunPod Jupyter Deployment - Reliable Method")
    print("="*70)
    print(f"Models: {', '.join(MODELS)}")
    print(f"Method: Jupyter Lab + manual command execution")
    print("Why: Complex automation keeps failing, this ALWAYS works")
    print()

    successful_pods = []

    for model in MODELS:
        pod_name = f"empathy-{model.replace('.', '-')}"
        print(f"\n🚀 Launching {model}...")

        pod = create_jupyter_pod(pod_name, model)
        if pod:
            successful_pods.append(pod)
        time.sleep(2)

    print("\n" + "="*70)
    print(f"✅ Launched {len(successful_pods)}/{len(MODELS)} pods")
    print("="*70)

    if successful_pods:
        print("\nPods:")
        for pod in successful_pods:
            print(f"  {pod['model']}: {pod['id']} ({pod['gpu']})")

        print_instructions(successful_pods)

        with open("pod_info.json", 'w') as f:
            json.dump(successful_pods, f, indent=2)


if __name__ == '__main__':
    main()
