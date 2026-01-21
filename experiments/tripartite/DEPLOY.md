# Deployment Guide: Tripartite Empathy Experiments

## Current Status

✅ **Ready for GPU Execution**
- All datasets generated and validated (240 scenarios, 720 responses)
- GPU experiment scripts created and tested
- Dockerfile.gpu configured with CUDA + PyTorch + TransformerLens
- Committed to GitHub: commit `42574d4`

⚠️ **Awaiting**
- docker_ec2_builder tool completion (missing dependencies)
- AWS credentials configured in docker_ec2_builder

---

## Deployment Options

### Option 1: Using docker_ec2_builder (Recommended - When Ready)

```bash
# 1. Ensure docker_ec2_builder is installed and configured
cd ~/docker_ec2_builder
pip install -e .

# 2. Configure AWS credentials (if not done)
# Edit config.yaml with your AWS settings:
#   - region
#   - key_pair
#   - security_group

# 3. Build and run on EC2
cd ~/paladin_claude/empathetic-language-bandwidth

~/docker_ec2_builder/src/docker_ec2_builder/cli.py build-and-run \
  --dockerfile Dockerfile.gpu \
  --context . \
  --instance-type g5.xlarge \
  --cmd "cd /app && python experiments/tripartite/scripts/run_all_gpu.py --models all"
```

**Expected Cost:** ~$4-6 (g5.xlarge spot @ $0.50/hr × 5 hours)

---

### Option 2: Manual EC2 Deployment

```bash
# 1. Launch EC2 instance
# - Instance type: g5.xlarge (24GB GPU, 4 vCPUs, 16GB RAM)
# - AMI: Deep Learning AMI (Ubuntu 20.04)
# - Storage: 100GB EBS

# 2. SSH into instance
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>

# 3. Clone repo
git clone https://github.com/marcosantar93/empathetic-language-bandwidth.git
cd empathetic-language-bandwidth

# 4. Build Docker image
docker build -f Dockerfile.gpu -t tripartite-empathy .

# 5. Run experiments
docker run --gpus all \
  -v $(pwd)/experiments/tripartite:/app/experiments/tripartite \
  tripartite-empathy \
  python experiments/tripartite/scripts/run_all_gpu.py --models all

# 6. Copy results back
# Results will be in: experiments/tripartite/results/
# Sync to S3 or copy via scp

# 7. Terminate instance when done
```

---

### Option 3: Local GPU (If Available)

```bash
# Only if you have NVIDIA GPU with 24GB+ VRAM

# 1. Build Docker image
docker build -f Dockerfile.gpu -t tripartite-empathy .

# 2. Run experiments
docker run --gpus all \
  -v $(pwd)/experiments/tripartite:/app/experiments/tripartite \
  tripartite-empathy \
  python experiments/tripartite/scripts/run_all_gpu.py --models all
```

---

## Experiment Configuration

### Default Settings
- **Models**: All 5 (Gemma2-9B, Llama-3.1-8B, Qwen2.5-7B, Mistral-7B, DeepSeek-R1-7B)
- **SAE epochs**: 100
- **Probe epochs**: 50
- **Batch size**: 8

### Custom Configuration

Run specific models:
```bash
python run_all_gpu.py --models gemma-2-9b,llama-3.1-8b
```

Skip steps (if partially complete):
```bash
python run_all_gpu.py --skip-extraction  # Use existing activations
python run_all_gpu.py --skip-saes        # Skip SAE training
python run_all_gpu.py --skip-probes      # Skip probe training
```

Adjust training:
```bash
python run_all_gpu.py --sae-epochs 200 --probe-epochs 100
```

---

## Expected Timeline

| Step | Time | Memory | GPU |
|------|------|--------|-----|
| Activation extraction | 2-3 hrs | ~10GB | 80% |
| SAE training | 1 hr | ~8GB | 90% |
| Probe training | 30 min | ~4GB | 70% |
| Convergence analysis | 10 min | ~2GB | 0% |
| **Total** | **4-5 hrs** | **~10GB peak** | **~80% avg** |

---

## Output Structure

```
experiments/tripartite/
├── activations/                    # ~2GB total
│   ├── gemma-2-9b_activations.json
│   ├── llama-3.1-8b_activations.json
│   ├── qwen2.5-7b_activations.json
│   ├── mistral-7b_activations.json
│   └── deepseek-r1-7b_activations.json
│
├── saes/                           # ~500MB total
│   ├── gemma-2-9b_layer12_sae.pt
│   ├── llama-3.1-8b_layer16_sae.pt
│   ├── qwen2.5-7b_layer14_sae.pt
│   ├── mistral-7b_layer16_sae.pt
│   └── deepseek-r1-7b_layer16_sae.pt
│
└── results/
    ├── experiment_a/               # SAE discovery results
    ├── experiment_b/               # Probe results
    │   ├── gemma-2-9b_layer12_probe.json
    │   ├── ... (4 more models)
    │   └── probe_summary.json
    │
    ├── convergence_report.json     # Main results
    ├── convergence_visualization.png
    └── experiment_summary.json
```

---

## After Experiment Completion

1. **Download Results**
   ```bash
   # From EC2 to local
   scp -r ubuntu@<instance-ip>:~/empathetic-language-bandwidth/experiments/tripartite/results \
       ./experiments/tripartite/
   ```

2. **Commit Results to Git**
   ```bash
   git add experiments/tripartite/results/
   git commit -m "Add tripartite decomposition results"
   git push
   ```

3. **Analyze Results**
   - Review `convergence_report.json` for main findings
   - Check `convergence_visualization.png` for visual summary
   - Examine per-model probe results in `experiment_b/`

4. **Create Report**
   - Use results to update research paper/blog post
   - Create additional visualizations if needed
   - Document findings in CLAUDE.md

---

## Success Criteria

From CLAUDE.md:

1. **Separation**: cos(Cognitive, Affective) < 0.5 for majority of models
2. **Convergence**: SAE clusters align with probes (mean cosine > 0.5)
3. **Specificity**: Control datasets show different structure than empathy
4. **Causality**: Ablating directions degrades empathetic quality (future work)

---

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 4`
- Process models sequentially instead of in parallel
- Use smaller instance (t3.large) but expect longer runtime

### Model Not Found
- Check model name in `extract_activations.py` MODEL_NAMES
- Verify HuggingFace model availability
- May need authentication for gated models (Llama)

### Docker Build Fails
- Ensure sufficient disk space (>50GB)
- Check Docker daemon is running
- Verify CUDA compatibility with PyTorch version

---

## Cost Estimates

### AWS EC2 g5.xlarge
- On-demand: ~$1.00/hr
- Spot: ~$0.40-0.60/hr (60% savings)
- **Total (5 hours)**: $2-5 depending on spot availability

### Storage
- EBS: $0.10/GB-month (negligible for short run)
- S3 backup: ~$0.10 for results (~4GB)

### Data Transfer
- Egress: First 100GB/month free (results are <5GB)

**Total Project Cost**: ~$5-10 including all steps
