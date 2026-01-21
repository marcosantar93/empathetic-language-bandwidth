#!/usr/bin/env python3
"""
Deploy tripartite empathy experiments to RunPod with parallel execution.

Uses RunPod API to:
1. Create Docker image with HF authentication
2. Launch one pod per model for parallel processing
3. Monitor progress and download results
4. Terminate pods when complete

Usage:
    python runpod_deploy.py --models all
    python runpod_deploy.py --models llama-3.1-8b,qwen2.5-7b --gpu-type "NVIDIA RTX 4090"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
if not RUNPOD_API_KEY:
    print("Error: RUNPOD_API_KEY environment variable not set")
    sys.exit(1)

HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN:
    print("Warning: HF_TOKEN not set. Gated models will fail.")

RUNPOD_API_BASE = "https://api.runpod.io/graphql"

# Model configurations
MODELS = {
    'llama-3.1-8b': {'vram_gb': 20, 'batch_size': 4},
    'qwen2.5-7b': {'vram_gb': 18, 'batch_size': 4},
    'mistral-7b': {'vram_gb': 18, 'batch_size': 4},
    'llama-3-8b': {'vram_gb': 20, 'batch_size': 4},
}

# GPU types (prices approximate, per hour)
GPU_TYPES = {
    'NVIDIA RTX 4090': {'vram': 24, 'price': 0.44},
    'NVIDIA RTX A5000': {'vram': 24, 'price': 0.45},
    'NVIDIA A40': {'vram': 48, 'price': 0.79},
}


def runpod_query(query: str, variables: Optional[Dict] = None) -> Dict:
    """Execute GraphQL query against RunPod API."""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': RUNPOD_API_KEY
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
    docker_image: str,
    gpu_type: str,
    gpu_count: int = 1,
    volume_size: int = 50,
    env_vars: Optional[Dict[str, str]] = None
) -> str:
    """Create a RunPod pod."""

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
            'imageName': docker_image,
            'gpuTypeId': gpu_type,
            'cloudType': 'ALL',
            'gpuCount': gpu_count,
            'volumeInGb': volume_size,
            'containerDiskInGb': volume_size,
            'dockerArgs': '',
            'ports': '8888/http',
            'volumeMountPath': '/workspace',
            'env': [{'key': k, 'value': v} for k, v in (env_vars or {}).items()]
        }
    }

    result = runpod_query(query, variables)
    pod = result['podFindAndDeployOnDemand']

    print(f"Created pod: {pod['id']} ({pod['name']}) - {pod['machine']['gpuDisplayName']}")
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

    variables = {
        'input': {'podId': pod_id}
    }

    result = runpod_query(query, variables)
    return result['pod']


def terminate_pod(pod_id: str):
    """Terminate a pod."""
    query = """
    mutation TerminatePod($input: PodTerminateInput!) {
        podTerminate(input: $input)
    }
    """

    variables = {
        'input': {'podId': pod_id}
    }

    runpod_query(query, variables)
    print(f"Terminated pod: {pod_id}")


def build_and_push_docker_image(tag: str = "latest") -> str:
    """Build Docker image with HF token and push to Docker Hub."""
    import subprocess

    dockerhub_username = os.environ.get('DOCKERHUB_USERNAME', 'marcosantar')
    image_name = f"{dockerhub_username}/empathetic-language-bandwidth"
    full_tag = f"{image_name}:{tag}"

    print(f"\n{'='*70}")
    print("Building Docker image with HuggingFace authentication")
    print(f"{'='*70}")

    # Build with HF token
    build_cmd = [
        'docker', 'build',
        '-f', 'Dockerfile.gpu',
        '--build-arg', f'HF_TOKEN={HF_TOKEN}',
        '-t', full_tag,
        '.'
    ]

    print(f"Command: {' '.join(build_cmd[:7])}... (HF_TOKEN hidden)")
    result = subprocess.run(build_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)

    print("✓ Docker build complete")

    # Push to Docker Hub
    print(f"\nPushing to Docker Hub: {full_tag}")
    push_cmd = ['docker', 'push', full_tag]
    result = subprocess.run(push_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Push failed:\n{result.stderr}")
        sys.exit(1)

    print(f"✓ Pushed to Docker Hub: {full_tag}")
    return full_tag


def deploy_parallel_experiments(
    models: List[str],
    docker_image: str,
    gpu_type: str = "NVIDIA RTX 4090"
) -> Dict[str, str]:
    """Deploy one pod per model for parallel execution."""

    print(f"\n{'='*70}")
    print("Deploying Parallel Experiments to RunPod")
    print(f"{'='*70}")
    print(f"Models: {', '.join(models)}")
    print(f"GPU Type: {gpu_type}")
    print()

    pod_ids = {}

    for model in models:
        pod_name = f"empathy-{model.replace('.', '-')}"

        # Environment variables
        env_vars = {
            'MODEL': model,
            'HF_TOKEN': HF_TOKEN,
            'BATCH_SIZE': str(MODELS[model]['batch_size'])
        }

        # Docker args to run single model
        docker_args = f"python experiments/tripartite/scripts/run_all_gpu.py --models {model}"

        try:
            pod_id = create_pod(
                name=pod_name,
                docker_image=docker_image,
                gpu_type=gpu_type,
                gpu_count=1,
                volume_size=50,
                env_vars=env_vars
            )
            pod_ids[model] = pod_id
            time.sleep(2)  # Rate limiting
        except Exception as e:
            print(f"✗ Failed to create pod for {model}: {e}")

    return pod_ids


def monitor_pods(pod_ids: Dict[str, str], check_interval: int = 60):
    """Monitor pod execution and download results when complete."""

    print(f"\n{'='*70}")
    print("Monitoring Experiment Progress")
    print(f"{'='*70}")
    print(f"Checking every {check_interval} seconds")
    print()

    completed = set()

    while len(completed) < len(pod_ids):
        for model, pod_id in pod_ids.items():
            if model in completed:
                continue

            try:
                status = get_pod_status(pod_id)
                uptime = status.get('runtime', {}).get('uptimeInSeconds', 0)

                print(f"[{model}] Pod {pod_id}: {uptime}s uptime")

                # Check if complete (would need to implement result checking)
                # For now, just monitor uptime

            except Exception as e:
                print(f"[{model}] Error checking status: {e}")

        time.sleep(check_interval)

    print("\n✓ All experiments complete!")


def cleanup_pods(pod_ids: Dict[str, str]):
    """Terminate all pods."""
    print(f"\n{'='*70}")
    print("Cleaning Up Pods")
    print(f"{'='*70}")

    for model, pod_id in pod_ids.items():
        try:
            terminate_pod(pod_id)
        except Exception as e:
            print(f"✗ Failed to terminate {model} pod: {e}")


def main():
    parser = argparse.ArgumentParser(description='Deploy experiments to RunPod')
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated model names or "all"')
    parser.add_argument('--gpu-type', type=str, default='NVIDIA RTX 4090',
                        choices=list(GPU_TYPES.keys()),
                        help='GPU type to use')
    parser.add_argument('--build-only', action='store_true',
                        help='Only build and push Docker image')
    parser.add_argument('--skip-build', action='store_true',
                        help='Skip Docker build, use existing image')
    parser.add_argument('--image-tag', type=str, default='latest',
                        help='Docker image tag')
    parser.add_argument('--monitor-interval', type=int, default=60,
                        help='Pod monitoring interval in seconds')

    args = parser.parse_args()

    # Parse models
    if args.models == 'all':
        models = list(MODELS.keys())
    else:
        models = [m.strip() for m in args.models.split(',')]
        invalid = [m for m in models if m not in MODELS]
        if invalid:
            print(f"Error: Invalid model(s): {', '.join(invalid)}")
            print(f"Valid models: {', '.join(MODELS.keys())}")
            sys.exit(1)

    # Build Docker image
    if not args.skip_build:
        docker_image = build_and_push_docker_image(args.image_tag)
    else:
        dockerhub_username = os.environ.get('DOCKERHUB_USERNAME', 'marcosantar')
        docker_image = f"{dockerhub_username}/empathetic-language-bandwidth:{args.image_tag}"
        print(f"Using existing image: {docker_image}")

    if args.build_only:
        print("\n✓ Build complete. Use --skip-build to deploy.")
        return

    # Deploy pods
    pod_ids = deploy_parallel_experiments(models, docker_image, args.gpu_type)

    if not pod_ids:
        print("✗ No pods created")
        sys.exit(1)

    print(f"\n✓ Created {len(pod_ids)} pods")
    print("\nPod IDs:")
    for model, pod_id in pod_ids.items():
        print(f"  {model}: {pod_id}")

    # Monitor (optional - for now just show IDs and user can monitor manually)
    print(f"\nMonitor your pods at: https://www.runpod.io/console/pods")
    print("\nTo terminate all pods, run:")
    print(f"  python runpod_deploy.py --cleanup {','.join(pod_ids.values())}")


if __name__ == '__main__':
    main()
