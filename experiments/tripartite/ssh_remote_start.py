#!/usr/bin/env python3
"""
Remote experiment starter via SSH
Requires SSH key to be added to RunPod account first
"""
import os
import subprocess
import json
import time

# Load pod info
with open("empathy_pods.json", 'r') as f:
    pods = json.load(f)

# Experiment commands
SETUP_CMD = """
set -e
cd /workspace
if [ ! -d "empathetic-language-bandwidth" ]; then
    git clone https://github.com/marcosantar93/empathetic-language-bandwidth.git
fi
cd empathetic-language-bandwidth
git pull
pip install -q --no-cache-dir -r requirements-gpu.txt
"""

def start_experiment_via_ssh(pod):
    """Start experiment on a pod via SSH"""
    ssh_info = pod.get('ssh')
    if not ssh_info or not ssh_info.get('ip'):
        print(f"  ⚠️  SSH not available yet for {pod['model']}")
        return False

    ip = ssh_info['ip']
    port = ssh_info['port']
    model = pod['model']

    print(f"\n{'='*70}")
    print(f"Starting {model} on {ip}:{port}")
    print('='*70)

    # Test connection first
    test_cmd = ['ssh', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=no',
                '-p', str(port), f'root@{ip}', 'echo "Connection successful"']

    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print(f"  ❌ SSH connection failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ❌ SSH connection timeout")
        return False

    print(f"  ✅ SSH connection successful")

    # Run setup
    print(f"  📦 Setting up environment...")
    setup_result = subprocess.run(
        ['ssh', '-p', str(port), f'root@{ip}', SETUP_CMD],
        capture_output=True, text=True, timeout=600
    )

    if setup_result.returncode != 0:
        print(f"  ❌ Setup failed: {setup_result.stderr}")
        return False

    print(f"  ✅ Setup complete")

    # Start experiment (use bash -c with proper daemonization)
    print(f"  🚀 Starting experiment...")
    exp_cmd = f"bash -c 'cd /workspace/empathetic-language-bandwidth && python experiments/tripartite/scripts/run_all_gpu.py --models {model} --batch-size 2 > experiment.log 2>&1 &' &"

    try:
        exp_result = subprocess.run(
            ['ssh', '-f', '-p', str(port), f'root@{ip}', exp_cmd],
            capture_output=True, text=True, timeout=5
        )
        print(f"  ✅ Experiment launch command sent!")
    except subprocess.TimeoutExpired:
        print(f"  ✅ Experiment started (background process)")
        pass

    # Give it a moment then check
    time.sleep(5)
    check_cmd = "ps aux | grep 'run_all_gpu' | grep -v grep | wc -l"
    try:
        check_result = subprocess.run(
            ['ssh', '-p', str(port), f'root@{ip}', check_cmd],
            capture_output=True, text=True, timeout=5
        )
        count = int(check_result.stdout.strip())
        if count > 0:
            print(f"  ✅ Process confirmed running ({count} process found)")
            return True
        else:
            print(f"  ⚠️  Process not found yet (may still be starting)")
            return True
    except Exception as e:
        print(f"  ⚠️  Could not verify process: {e}")
        return True

if __name__ == "__main__":
    print("="*70)
    print("Remote Experiment Starter - SSH Method")
    print("="*70)

    # Check SSH key
    print("\n1. Checking SSH key...")
    local_key = os.path.expanduser("~/.ssh/id_ed25519.pub")
    if os.path.exists(local_key):
        with open(local_key) as f:
            key = f.read().strip()
        print(f"  ✅ Local SSH key found: {key[:50]}...")
    else:
        print(f"  ❌ No SSH key found at {local_key}")
        print(f"  Run: ssh-keygen -t ed25519 -C 'your_email@example.com'")
        exit(1)

    # Filter pods with SSH info
    ssh_ready_pods = [p for p in pods if p.get('ssh') and p['ssh'].get('ip')]

    print(f"\n2. Found {len(ssh_ready_pods)}/{len(pods)} pods with SSH ready")

    if len(ssh_ready_pods) == 0:
        print("\n  ⚠️  No pods have SSH ready yet. Wait a bit longer or use Jupyter.")
        exit(1)

    print("\n3. Testing SSH connections...")

    success_count = 0
    for pod in ssh_ready_pods:
        if start_experiment_via_ssh(pod):
            success_count += 1
        time.sleep(2)

    print("\n" + "="*70)
    print(f"✅ Started {success_count}/{len(ssh_ready_pods)} experiments successfully")
    print("="*70)

    if success_count < len(ssh_ready_pods):
        print("\nFor failed pods, you can still use Jupyter Lab:")
        print("See START_INSTRUCTIONS.md for manual commands")
