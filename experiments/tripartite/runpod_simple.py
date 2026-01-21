#!/usr/bin/env python3
"""
Simplified RunPod deployment using PyTorch base image + git clone.

No Docker build required - pulls code directly from GitHub.

Usage:
    python runpod_simple.py --models llama-3.1-8b
    python runpod_simple.py --models all --parallel
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import requests


RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
HF_TOKEN = os.environ.get('HF_TOKEN')
GITHUB_REPO = "marcosantar93/empathetic-language-bandwidth"
GITHUB_BRANCH = "main"

if not RUNPOD_API_KEY:
    print("Error: RUNPOD_API_KEY not set")
    sys.exit(1)

if not HF_TOKEN:
    print("Warning: HF_TOKEN not set. Gated models will fail.")

RUNPOD_API_BASE = "https://api.runpod.io/graphql"


def runpod_query(query: str, variables: Optional[Dict] = None) -> Dict:
    """Execute GraphQL query."""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {RUNPOD_API_KEY}'  # Fixed: needs Bearer prefix
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


def create_pod(
    name: str,
    model: str,
    gpu_type: str = "NVIDIA RTX 4090",
    volume_size: int = 50
) -> str:
    """Create RunPod pod with startup script."""

    # Startup script to clone repo, install deps, run experiments
    # Note: Using bash -c with proper quoting for the entire script
    docker_args = f"""bash -c 'set -e && \\
    echo "=== Setting up environment ===" && \\
    pip install -q -U huggingface_hub[cli] && \\
    huggingface-cli login --token {HF_TOKEN} && \\
    echo "=== Cloning repository ===" && \\
    cd /workspace && \\
    git clone -q https://github.com/{GITHUB_REPO}.git && \\
    cd empathetic-language-bandwidth && \\
    git checkout {GITHUB_BRANCH} && \\
    echo "=== Installing dependencies ===" && \\
    pip install -q --no-cache-dir -r requirements-gpu.txt && \\
    echo "=== Starting experiments ===" && \\
    python experiments/tripartite/scripts/run_all_gpu.py --models {model} --batch-size 2 && \\
    echo "=== Experiments complete ===" && \\
    echo "Results in: /workspace/empathetic-language-bandwidth/experiments/tripartite/results/" || \\
    (echo "=== Experiment failed ===" && tail -100 /workspace/*.log && exit 1)
    '"""

    query = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            name
            imageName
            machineId
            machine {
                gpuDisplayName
            }
        }
    }
    """

    variables = {
        'input': {
            'name': name,
            'imageName': 'pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime',
            'gpuTypeId': gpu_type,
            'cloudType': 'COMMUNITY',  # Changed from ALL to COMMUNITY for cheaper pricing
            'gpuCount': 1,
            'volumeInGb': volume_size,
            'containerDiskInGb': volume_size,
            'dockerArgs': docker_args,
            'ports': '8888/http',
            'volumeMountPath': '/workspace',
            'env': [
                {'key': 'JUPYTER_PASSWORD', 'value': 'runpod'},  # CRITICAL: Required for Jupyter
                {'key': 'HF_TOKEN', 'value': HF_TOKEN},
                {'key': 'MODEL', 'value': model},
                {'key': 'PYTHONUNBUFFERED', 'value': '1'}  # Better logging
            ]
        }
    }

    result = runpod_query(query, variables)
    pod = result['podFindAndDeployOnDemand']

    print(f"✓ Created pod: {pod['id']} ({pod['name']}) - {pod['machine']['gpuDisplayName']}")
    return pod['id']


def get_pod_status(pod_id: str) -> Dict:
    """Get pod status."""
    query = """
    query GetPod($input: PodFilter!) {
        pod(input: $input) {
            id
            name
            runtime {
                uptimeInSeconds
                ports {
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                }
            }
            machine {
                gpuDisplayName
            }
        }
    }
    """

    variables = {'input': {'podId': pod_id}}
    result = runpod_query(query, variables)
    return result['pod']


def stop_pod(pod_id: str):
    """Stop pod."""
    query = """
    mutation StopPod($input: PodStopInput!) {
        podStop(input: $input) {
            id
        }
    }
    """
    variables = {'input': {'podId': pod_id}}
    runpod_query(query, variables)
    print(f"✓ Stopped pod: {pod_id}")


def terminate_pod(pod_id: str):
    """Terminate pod."""
    query = """
    mutation TerminatePod($input: PodTerminateInput!) {
        podTerminate(input: $input)
    }
    """
    variables = {'input': {'podId': pod_id}}
    runpod_query(query, variables)
    print(f"✓ Terminated pod: {pod_id}")


def main():
    parser = argparse.ArgumentParser(description='Simple RunPod deployment')
    parser.add_argument('--models', type=str, required=True,
                        help='Comma-separated model names or "all"')
    parser.add_argument('--parallel', action='store_true',
                        help='Run models in parallel (one pod per model)')
    parser.add_argument('--gpu-type', type=str, default='NVIDIA RTX 4090',
                        help='GPU type')
    parser.add_argument('--terminate', type=str,
                        help='Terminate pod by ID')
    parser.add_argument('--status', type=str,
                        help='Get status of pod by ID')

    args = parser.parse_args()

    # Terminate pod
    if args.terminate:
        terminate_pod(args.terminate)
        return

    # Get status
    if args.status:
        status = get_pod_status(args.status)
        print(json.dumps(status, indent=2))
        return

    # Parse models
    all_models = ['llama-3.1-8b', 'qwen2.5-7b', 'mistral-7b', 'llama-3-8b']
    if args.models == 'all':
        models = all_models
    else:
        models = [m.strip() for m in args.models.split(',')]

    print(f"\n{'='*70}")
    print("RunPod Deployment - Tripartite Empathy Experiments")
    print(f"{'='*70}")
    print(f"Models: {', '.join(models)}")
    print(f"GPU Type: {args.gpu_type}")
    print(f"Parallel: {args.parallel}")
    print()

    pod_ids = {}

    if args.parallel:
        # Create one pod per model
        for model in models:
            pod_name = f"empathy-{model.replace('.', '-')}"
            try:
                pod_id = create_pod(pod_name, model, args.gpu_type)
                pod_ids[model] = pod_id
                time.sleep(2)  # Rate limit
            except Exception as e:
                print(f"✗ Failed to create pod for {model}: {e}")
    else:
        # Create single pod for all models (sequential)
        pod_name = "empathy-all"
        models_str = ','.join(models)
        try:
            pod_id = create_pod(pod_name, models_str, args.gpu_type)
            pod_ids['all'] = pod_id
        except Exception as e:
            print(f"✗ Failed to create pod: {e}")
            sys.exit(1)

    print(f"\n{'='*70}")
    print(f"✓ Created {len(pod_ids)} pod(s)")
    print(f"{'='*70}")

    for model, pod_id in pod_ids.items():
        print(f"{model}: {pod_id}")

    print(f"\nMonitor at: https://www.runpod.io/console/pods")
    print(f"\nTo terminate:")
    for pod_id in pod_ids.values():
        print(f"  python runpod_simple.py --terminate {pod_id}")


if __name__ == '__main__':
    main()
