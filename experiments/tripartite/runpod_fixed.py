#!/usr/bin/env python3
"""
RunPod Launcher - Fixed Version

Based on lessons from previous RunPod debugging:
1. NO complex dockerArgs - let container start normally
2. Use default Jupyter behavior
3. Create a startup script that gets written to the pod
4. Run experiments via simple command

Usage:
    python runpod_fixed.py
"""

import os
import sys
import json
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

RUNPOD_API_BASE = "https://api.runpod.io/graphql"

GPU_TYPES = [
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
]

MODELS = ['llama-3.1-8b', 'qwen2.5-7b', 'mistral-7b', 'llama-3-8b']


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


def create_pod_simple(name: str, model: str) -> Optional[Dict]:
    """
    Create pod with MINIMAL dockerArgs.
    Let container start normally, write experiment script to volume.
    """

    # MINIMAL startup - just keep container alive
    # Container will start Jupyter automatically with JUPYTER_PASSWORD
    docker_args = "sleep infinity"

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
                    'dockerArgs': docker_args,  # SIMPLE!
                    'ports': '8888/http',
                    'volumeMountPath': '/workspace',
                    'env': [
                        {'key': 'JUPYTER_PASSWORD', 'value': 'runpod'},  # Critical!
                        {'key': 'HF_TOKEN', 'value': HF_TOKEN},
                        {'key': 'MODEL', 'value': model},
                        {'key': 'PYTHONUNBUFFERED', 'value': '1'},
                        {'key': 'GITHUB_REPO', 'value': GITHUB_REPO},
                        {'key': 'GITHUB_BRANCH', 'value': GITHUB_BRANCH},
                    ]
                }
            }

            print(f"   Trying {gpu_type}...")
            result = runpod_query(query, variables)
            pod = result['podFindAndDeployOnDemand']

            print(f"   ✅ Pod created: {pod['id']} ({pod['machine']['gpuDisplayName']})")

            # Now write startup script to pod
            time.sleep(10)  # Let pod initialize
            write_startup_script(pod['id'], model)

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
            else:
                print(f"   ❌ Error: {e}")
                continue

    return None


def write_startup_script(pod_id: str, model: str):
    """
    Write experiment startup script to pod using GraphQL exec.
    This runs AFTER pod is up, so it's more reliable.
    """
    print(f"   Writing startup script to pod...")

    # Create startup script content
    script_content = f'''#!/bin/bash
set -e
cd /workspace

echo "=== Installing dependencies ===" | tee -a experiment.log
pip install -q huggingface_hub[cli] 2>&1 | tee -a experiment.log
huggingface-cli login --token $HF_TOKEN 2>&1 | tee -a experiment.log

echo "=== Cloning repository ===" | tee -a experiment.log
if [ ! -d "empathetic-language-bandwidth" ]; then
    git clone https://github.com/$GITHUB_REPO.git 2>&1 | tee -a experiment.log
fi
cd empathetic-language-bandwidth
git checkout $GITHUB_BRANCH 2>&1 | tee -a experiment.log

echo "=== Installing Python dependencies ===" | tee -a experiment.log
pip install -q --no-cache-dir -r requirements-gpu.txt 2>&1 | tee -a experiment.log

echo "=== Starting experiments ===" | tee -a /workspace/experiment.log
python experiments/tripartite/scripts/run_all_gpu.py --models {model} --batch-size 2 2>&1 | tee -a /workspace/experiment.log

echo "=== Experiments complete ===" | tee -a /workspace/experiment.log
'''

    # Write script to file
    exec_query = """
    mutation {{
        podExecuteCommand(input: {{
            podId: "{pod_id}"
            command: "cat > /workspace/run_experiment.sh << 'SCRIPT_EOF'\\n{script}\\nSCRIPT_EOF && chmod +x /workspace/run_experiment.sh"
        }}) {{
            output
        }}
    }}
    """.format(pod_id=pod_id, script=script_content.replace('\n', '\\n'))

    try:
        runpod_query(exec_query)
        print(f"   ✓ Script written")

        # Start the script in background
        start_query = f"""
        mutation {{
            podExecuteCommand(input: {{
                podId: "{pod_id}"
                command: "nohup /workspace/run_experiment.sh > /workspace/experiment.log 2>&1 &"
            }}) {{
                output
            }}
        }}
        """
        runpod_query(start_query)
        print(f"   ✓ Experiment started in background")

    except Exception as e:
        print(f"   ⚠️ Could not write script: {e}")
        print(f"   Manual start: SSH to pod and run commands manually")


def main():
    print("\n" + "="*70)
    print("RunPod Deployment - FIXED VERSION")
    print("="*70)
    print(f"Models: {', '.join(MODELS)}")
    print(f"Approach: Minimal dockerArgs + post-startup script")
    print()

    successful_pods = []

    for model in MODELS:
        pod_name = f"empathy-{model.replace('.', '-')}"
        print(f"\n🚀 Launching {model}...")

        pod = create_pod_simple(pod_name, model)

        if pod:
            successful_pods.append(pod)
        else:
            print(f"   ❌ Failed to launch {model}")

        time.sleep(3)

    print("\n" + "="*70)
    print("Launch Summary")
    print("="*70)
    print(f"✅ Successful: {len(successful_pods)}/{len(MODELS)}")

    if successful_pods:
        print("\nRunning pods:")
        for pod in successful_pods:
            print(f"  {pod['model']}: {pod['id']}")

        print(f"\nJupyter Lab: https://www.runpod.io/console/pods")
        print(f"Password: runpod")
        print(f"\nCheck logs: Open pod terminal and run: tail -f /workspace/experiment.log")

    if successful_pods:
        with open("pod_info.json", 'w') as f:
            json.dump(successful_pods, f, indent=2)


if __name__ == '__main__':
    main()
