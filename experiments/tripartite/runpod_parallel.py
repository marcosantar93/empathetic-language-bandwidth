#!/usr/bin/env python3
"""
RunPod Parallel Launcher with GPU Fallback

Tries multiple GPU types for availability and launches models in parallel.
Based on lessons from previous RunPod projects.

Usage:
    python runpod_parallel.py                    # Launch all 4 models in parallel
    python runpod_parallel.py --status           # Check pod statuses
    python runpod_parallel.py --terminate-all    # Terminate all pods
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional

import requests

# Configuration
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

# GPU types to try (in order of preference)
GPU_TYPES = [
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A4000",
]

MODELS = [
    'llama-3.1-8b',
    'qwen2.5-7b',
    'mistral-7b',
    'llama-3-8b'
]

def runpod_query(query: str, variables: Optional[Dict] = None) -> Dict:
    """Execute GraphQL query."""
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


def create_pod_with_fallback(name: str, model: str) -> Optional[Dict]:
    """Try creating pod with multiple GPU types."""

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
    echo "Results saved to: /workspace/empathetic-language-bandwidth/experiments/tripartite/results/" || \\
    (echo "=== Experiment failed ===" && ls -la /workspace/*.log 2>/dev/null && exit 1)
    '"""

    query = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            name
            imageName
            machine {
                gpuDisplayName
            }
        }
    }
    """

    # Try each GPU type
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
                    'dockerArgs': docker_args,
                    'ports': '8888/http',
                    'volumeMountPath': '/workspace',
                    'env': [
                        {'key': 'JUPYTER_PASSWORD', 'value': 'runpod'},
                        {'key': 'HF_TOKEN', 'value': HF_TOKEN},
                        {'key': 'MODEL', 'value': model},
                        {'key': 'PYTHONUNBUFFERED', 'value': '1'}
                    ]
                }
            }

            print(f"   Trying {gpu_type}...")
            result = runpod_query(query, variables)
            pod = result['podFindAndDeployOnDemand']

            print(f"   ✅ Success! Pod {pod['id']} ({pod['machine']['gpuDisplayName']})")
            return {
                'id': pod['id'],
                'name': pod['name'],
                'gpu': pod['machine']['gpuDisplayName'],
                'model': model
            }

        except Exception as e:
            error_msg = str(e)
            if "no longer any instances available" in error_msg.lower():
                print(f"   ⚠️ {gpu_type} not available")
                continue
            else:
                print(f"   ❌ Error with {gpu_type}: {e}")
                continue

    return None


def get_all_pods() -> List[Dict]:
    """Get all running pods."""
    query = """
    query {
        myself {
            pods {
                id
                name
                desiredStatus
                runtime {
                    uptimeInSeconds
                }
                machine {
                    gpuDisplayName
                }
            }
        }
    }
    """

    result = runpod_query(query)
    return result.get('myself', {}).get('pods', [])


def terminate_pod(pod_id: str):
    """Terminate a pod."""
    query = """
    mutation TerminatePod($input: PodTerminateInput!) {
        podTerminate(input: $input)
    }
    """
    variables = {'input': {'podId': pod_id}}
    runpod_query(query, variables)
    print(f"✓ Terminated {pod_id}")


def main():
    parser = argparse.ArgumentParser(description='RunPod Parallel Launcher')
    parser.add_argument('--status', action='store_true',
                        help='Show status of all pods')
    parser.add_argument('--terminate-all', action='store_true',
                        help='Terminate all pods')

    args = parser.parse_args()

    if args.status:
        print("\n" + "="*70)
        print("Pod Status")
        print("="*70)
        pods = get_all_pods()
        if not pods:
            print("No pods running")
        else:
            for pod in pods:
                status = pod.get('desiredStatus', 'UNKNOWN')
                uptime = pod.get('runtime', {}).get('uptimeInSeconds', 0)
                gpu = pod.get('machine', {}).get('gpuDisplayName', 'Unknown GPU')
                print(f"{pod['name']}: {status} ({uptime}s) - {gpu}")
                print(f"  Pod ID: {pod['id']}")
        return

    if args.terminate_all:
        print("\n" + "="*70)
        print("Terminating All Pods")
        print("="*70)
        pods = get_all_pods()
        for pod in pods:
            if 'empathy-' in pod['name']:
                terminate_pod(pod['id'])
        return

    # Launch all models in parallel
    print("\n" + "="*70)
    print("RunPod Parallel Deployment")
    print("="*70)
    print(f"Models: {', '.join(MODELS)}")
    print(f"Strategy: 1 pod per model, GPU fallback enabled")
    print()

    successful_pods = []
    failed_models = []

    for model in MODELS:
        pod_name = f"empathy-{model.replace('.', '-')}"
        print(f"\n🚀 Launching pod for {model}...")

        pod = create_pod_with_fallback(pod_name, model)

        if pod:
            successful_pods.append(pod)
        else:
            print(f"   ❌ Failed to launch {model} (no GPUs available)")
            failed_models.append(model)

        time.sleep(2)  # Rate limiting

    # Summary
    print("\n" + "="*70)
    print("Launch Summary")
    print("="*70)
    print(f"✅ Successful: {len(successful_pods)}/{len(MODELS)}")
    if successful_pods:
        print("\nRunning pods:")
        for pod in successful_pods:
            print(f"  {pod['model']}: {pod['id']} ({pod['gpu']})")

    if failed_models:
        print(f"\n❌ Failed: {', '.join(failed_models)}")

    print(f"\nMonitor at: https://www.runpod.io/console/pods")
    print(f"\nCheck status: python runpod_parallel.py --status")
    print(f"Terminate all: python runpod_parallel.py --terminate-all")

    # Save pod info
    if successful_pods:
        pod_info_file = "pod_info.json"
        with open(pod_info_file, 'w') as f:
            json.dump(successful_pods, f, indent=2)
        print(f"\n📊 Pod info saved to: {pod_info_file}")


if __name__ == '__main__':
    main()
