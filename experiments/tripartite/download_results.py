#!/usr/bin/env python3
"""Download results from RunPod experiments"""
import subprocess
import os

PODS = [
    {'name': 'qwen2.5-7b', 'ip': '190.111.198.202', 'port': 58996},
    {'name': 'mistral-7b', 'ip': '190.111.198.202', 'port': 13260},
]

# Create results directory
os.makedirs('results', exist_ok=True)

print("="*70)
print("Downloading Results from RunPod")
print("="*70)

for pod in PODS:
    print(f"\n📥 Downloading from {pod['name']}...")

    # Create model-specific directory
    model_dir = f"results/{pod['name']}"
    os.makedirs(model_dir, exist_ok=True)

    # Download results directory
    cmd = [
        'scp', '-r',
        '-o', 'ConnectTimeout=30',
        '-o', 'StrictHostKeyChecking=no',
        '-P', str(pod['port']),
        f"root@{pod['ip']}:/workspace/empathetic-language-bandwidth/experiments/tripartite/results/",
        model_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✅ Downloaded to {model_dir}/")

        # List downloaded files
        result_path = f"{model_dir}/results"
        if os.path.exists(result_path):
            for root, dirs, files in os.walk(result_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path)
                    rel_path = os.path.relpath(file_path, model_dir)
                    print(f"     - {rel_path} ({size:,} bytes)")
    else:
        print(f"  ❌ Download failed: {result.stderr}")

    # Also download experiment.log
    log_cmd = [
        'scp',
        '-o', 'ConnectTimeout=30',
        '-o', 'StrictHostKeyChecking=no',
        '-P', str(pod['port']),
        f"root@{pod['ip']}:/workspace/empathetic-language-bandwidth/experiment.log",
        f"{model_dir}/experiment.log"
    ]

    subprocess.run(log_cmd, capture_output=True)

print("\n" + "="*70)
print("✅ Download complete!")
print("="*70)
print(f"\nResults location: ./results/")
print("\nKey files to check:")
print("  - results/*/results/convergence_report.json")
print("  - results/*/results/convergence_visualization.png")
print("  - results/*/results/experiment_summary.json")
print("  - results/*/experiment.log")

print("\nNext step: python3 runpod_working_models.py --cleanup")
