#!/usr/bin/env python3
"""
RunPod Empathy Experiments Launcher
====================================

Based on proven ~/runpod_debug suite with JUPYTER_PASSWORD fix.
Launches pods for tripartite empathy decomposition experiments.

Usage:
    python3 runpod_empathy.py              # Launch all 4 models
    python3 runpod_empathy.py --status     # Check pod status
    python3 runpod_empathy.py --cleanup    # Terminate all pods
"""

import os
import sys
import requests
import time
import argparse
import json

# Configuration
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
RUNPOD_API_URL = "https://api.runpod.io/graphql"
DOCKER_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel"  # RunPod-specific image required!
SSH_PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAG3DOFBy6LQv+6O5XJiugLOguagD5SuuX5ZpBY1pGqp marcosantar93@gmail.com"

# Models to run
MODELS = ['llama-3.1-8b', 'qwen2.5-7b', 'mistral-7b', 'llama-3-8b']

# GPU options (proven order from runpod_debug)
GPU_OPTIONS = [
    "NVIDIA RTX A5000",         # $0.39-0.45/hr - Most available
    "NVIDIA GeForce RTX 3090",  # $0.17-0.22/hr - Best value
    "NVIDIA GeForce RTX 4090",  # $0.34/hr - Fastest
]


def graphql_query(query):
    """Execute GraphQL query"""
    if not RUNPOD_API_KEY:
        raise ValueError("RUNPOD_API_KEY not set. Get it at: https://www.runpod.io/console/user/settings")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }

    response = requests.post(RUNPOD_API_URL, json={"query": query}, headers=headers)
    result = response.json()

    if "errors" in result:
        raise RuntimeError(f"GraphQL error: {result['errors']}")

    return result["data"]


def create_empathy_pod(model_name):
    """
    Create pod for empathy experiment.

    CRITICAL: Empty dockerArgs + JUPYTER_PASSWORD (proven from runpod_debug)
    """

    pod_name = f"empathy-{model_name.replace('.', '-')}"

    for gpu_type in GPU_OPTIONS:
        query = f'''
        mutation {{
            podFindAndDeployOnDemand(
                input: {{
                    cloudType: COMMUNITY
                    gpuCount: 1
                    volumeInGb: 50
                    containerDiskInGb: 50
                    minVcpuCount: 8
                    minMemoryInGb: 32
                    gpuTypeId: "{gpu_type}"
                    name: "{pod_name}"
                    imageName: "{DOCKER_IMAGE}"
                    dockerArgs: ""
                    ports: "22/tcp,8888/http"
                    volumeMountPath: "/workspace"
                    env: [
                        {{key: "JUPYTER_PASSWORD", value: "runpod"}}
                        {{key: "PUBLIC_KEY", value: "{SSH_PUBLIC_KEY}"}}
                        {{key: "HF_TOKEN", value: "{HF_TOKEN}"}}
                        {{key: "MODEL", value: "{model_name}"}}
                        {{key: "PYTHONUNBUFFERED", value: "1"}}
                    ]
                }}
            ) {{
                id
                name
                costPerHr
                machine {{
                    gpuDisplayName
                }}
            }}
        }}
        '''

        try:
            print(f"   Trying {gpu_type}...")
            result = graphql_query(query)
            pod_data = result.get("podFindAndDeployOnDemand")

            if pod_data:
                pod_id = pod_data["id"]
                gpu_name = pod_data.get("machine", {}).get("gpuDisplayName", "Unknown")
                cost = pod_data.get("costPerHr", 0)

                print(f"   ✅ Success! Pod {pod_id} ({gpu_name}, ${cost:.2f}/hr)")
                return {
                    'id': pod_id,
                    'name': pod_name,
                    'model': model_name,
                    'gpu': gpu_name,
                    'cost_per_hr': cost
                }

        except Exception as e:
            if "no longer any instances available" in str(e).lower():
                print(f"   ⚠️ {gpu_type} not available")
                continue
            print(f"   ❌ Error: {e}")
            continue

    return None


def get_all_empathy_pods():
    """Get all pods with 'empathy' in name"""
    query = '''
    query {
        myself {
            pods {
                id
                name
                desiredStatus
                costPerHr
                runtime {
                    uptimeInSeconds
                    gpus {
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
                machine {
                    gpuDisplayName
                }
            }
        }
    }
    '''

    result = graphql_query(query)
    all_pods = result.get('myself', {}).get('pods', [])

    # Filter to empathy pods only
    return [p for p in all_pods if 'empathy' in p.get('name', '').lower()]


def terminate_pod(pod_id):
    """Terminate a pod"""
    query = f'''
    mutation {{
        podTerminate(input: {{podId: "{pod_id}"}})
    }}
    '''
    graphql_query(query)


def print_experiment_instructions(pods):
    """Print instructions for running experiments"""

    print("\n" + "="*70)
    print("HOW TO START EXPERIMENTS")
    print("="*70)
    print("\n1. Go to: https://www.runpod.io/console/pods")
    print("2. For EACH pod, click 'Connect' → 'Start Jupyter Lab'")
    print("3. Open a terminal in Jupyter Lab")
    print("4. Run these commands:\n")

    print("```bash")
    print("# Setup (run once per pod)")
    print("pip install -q huggingface_hub[cli]")
    print("huggingface-cli login --token $HF_TOKEN")
    print("cd /workspace")
    print("git clone https://github.com/marcosantar93/empathetic-language-bandwidth.git")
    print("cd empathetic-language-bandwidth")
    print("pip install -q --no-cache-dir -r requirements-gpu.txt")
    print()
    print("# Start experiment (replace $MODEL with your pod's model)")
    print("nohup python experiments/tripartite/scripts/run_all_gpu.py --models $MODEL --batch-size 2 > experiment.log 2>&1 &")
    print()
    print("# Monitor progress")
    print("tail -f experiment.log")
    print("```\n")

    print("Models for each pod:")
    for pod in pods:
        print(f"  {pod['name']}: {pod['model']}")

    print("\n" + "="*70)
    print("Expected timeline: 1.5-2 hours per model")
    print("GPU usage should show 80-100% when running")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='RunPod Empathy Experiments')
    parser.add_argument('--status', action='store_true', help='Check pod status')
    parser.add_argument('--cleanup', action='store_true', help='Terminate all empathy pods')
    args = parser.parse_args()

    if not RUNPOD_API_KEY:
        print("❌ RUNPOD_API_KEY not set")
        print("Get your key at: https://www.runpod.io/console/user/settings")
        print("\nThen run: export RUNPOD_API_KEY='rpa_YOUR_KEY'")
        sys.exit(1)

    # Status check
    if args.status:
        print("\n" + "="*70)
        print("Pod Status")
        print("="*70)

        pods = get_all_empathy_pods()
        if not pods:
            print("No empathy pods running")
        else:
            for pod in pods:
                status = pod.get('desiredStatus', 'UNKNOWN')
                runtime = pod.get('runtime') or {}
                uptime = runtime.get('uptimeInSeconds', 0)
                gpus = runtime.get('gpus') or []
                gpu_util = gpus[0].get('gpuUtilPercent', 0) if gpus else 0
                gpu_name = pod.get('machine', {}).get('gpuDisplayName', 'Unknown')
                cost = pod.get('costPerHr', 0)

                print(f"\n{pod['name']}: {status}")
                print(f"  Pod ID: {pod['id']}")
                print(f"  GPU: {gpu_name} (${cost:.2f}/hr)")
                print(f"  Uptime: {uptime}s")
                print(f"  GPU Usage: {gpu_util}%")
        return

    # Cleanup
    if args.cleanup:
        print("\n" + "="*70)
        print("Terminating All Empathy Pods")
        print("="*70)

        pods = get_all_empathy_pods()
        if not pods:
            print("No empathy pods to terminate")
        else:
            for pod in pods:
                print(f"Terminating {pod['name']} ({pod['id']})...")
                terminate_pod(pod['id'])
                print(f"  ✓ Terminated")
        return

    # Launch pods
    print("\n" + "="*70)
    print("RunPod Empathy Experiments - Proven Approach")
    print("="*70)
    print(f"Models: {', '.join(MODELS)}")
    print(f"Method: Empty dockerArgs + JUPYTER_PASSWORD (from ~/runpod_debug)")
    print()

    successful_pods = []

    for model in MODELS:
        print(f"\n🚀 Launching {model}...")
        pod = create_empathy_pod(model)

        if pod:
            successful_pods.append(pod)
        else:
            print(f"   ❌ Failed to launch {model}")

        time.sleep(2)  # Rate limiting

    # Summary
    print("\n" + "="*70)
    print(f"✅ Launched {len(successful_pods)}/{len(MODELS)} pods")
    print("="*70)

    if successful_pods:
        total_cost = sum(p['cost_per_hr'] for p in successful_pods)
        print(f"\nTotal cost: ${total_cost:.2f}/hour")
        print("\nPods:")
        for pod in successful_pods:
            print(f"  {pod['model']}: {pod['id']} ({pod['gpu']})")

        # Save pod info
        with open("empathy_pods.json", 'w') as f:
            json.dump(successful_pods, f, indent=2)
        print("\n📊 Pod info saved to: empathy_pods.json")

        # Print instructions
        print_experiment_instructions(successful_pods)

        print("\n" + "="*70)
        print("Commands:")
        print("  Check status: python3 runpod_empathy.py --status")
        print("  Terminate: python3 runpod_empathy.py --cleanup")
        print("="*70)


if __name__ == '__main__':
    main()
