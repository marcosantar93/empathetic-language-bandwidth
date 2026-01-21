# RunPod Parallel Deployment Guide

Deploy tripartite empathy experiments to RunPod with parallel execution for faster results.

## Prerequisites

1. **RunPod Account**: Sign up at https://www.runpod.io
2. **API Credentials**:
   ```bash
   export RUNPOD_API_KEY="your_runpod_api_key"
   export HF_TOKEN="your_huggingface_token"
   export DOCKERHUB_USERNAME="your_dockerhub_username"
   ```
   (These should already be in your ~/.api_credentials)

3. **HuggingFace Access**: Accept terms for gated models:
   - https://huggingface.co/meta-llama/Llama-3.1-8B
   - https://huggingface.co/meta-llama/Meta-Llama-3-8B

4. **Docker Hub**: Have a Docker Hub account for hosting images

## Deployment Steps

### Option A: Automatic Parallel Deployment (Recommended)

Deploy all 4 models in parallel on separate pods:

```bash
cd experiments/tripartite

# Build Docker image with HF auth and push to Docker Hub
python runpod_deploy.py --models all

# This will:
# 1. Build Docker image with HF_TOKEN embedded
# 2. Push to Docker Hub
# 3. Create 4 RunPod pods (one per model)
# 4. Start experiments in parallel
```

### Option B: Single Model Deployment

Test with one model first:

```bash
python runpod_deploy.py --models llama-3.1-8b
```

### Option C: Custom GPU Type

Use different GPU (e.g., A40 with 48GB VRAM for larger batches):

```bash
python runpod_deploy.py --models all --gpu-type "NVIDIA A40"
```

## Cost Comparison

### EC2 Sequential Execution
- Instance: g5.xlarge (24GB VRAM)
- Cost: $1.00/hour
- Duration: 4-5 hours (sequential)
- **Total: $4-5**

### RunPod Parallel Execution
- GPU: RTX 4090 (24GB VRAM)
- Cost: $0.44/hour per pod
- Duration: 1-2 hours (parallel)
- Pods: 4 simultaneous
- **Total: $1.76-3.52** (4 pods × $0.44/hr × 1-2 hrs)

**Savings: ~40% cheaper + 3x faster**

## GPU Options

| GPU Type | VRAM | Price/hr | Best For |
|----------|------|----------|----------|
| NVIDIA RTX 4090 | 24GB | $0.44 | Standard models (7B-8B) |
| NVIDIA RTX A5000 | 24GB | $0.45 | Standard models (7B-8B) |
| NVIDIA A40 | 48GB | $0.79 | Larger models or bigger batches |

## Monitoring

### Web Dashboard
Monitor pods at: https://www.runpod.io/console/pods

### CLI Status
```bash
# List all running pods
python runpod_deploy.py --list

# Get status of specific pod
python runpod_deploy.py --status <pod_id>
```

## Downloading Results

Results are stored in the Docker container at `/app/experiments/tripartite/results/`.

### Option 1: RunPod Web Terminal
1. Go to https://www.runpod.io/console/pods
2. Click "Connect" on your pod
3. Use the web terminal to browse results
4. Download via the file browser

### Option 2: SSH/SFTP
```bash
# Get SSH connection string from pod details
ssh root@<pod-ip> -p <port>

# Navigate to results
cd /app/experiments/tripartite/results

# Download results
scp -P <port> -r root@<pod-ip>:/app/experiments/tripartite/results ./results_from_runpod/
```

### Option 3: Mount to Cloud Storage
Update Dockerfile to sync results to S3/GCS during execution.

## Cleanup

### Automatic Cleanup
Pods will auto-terminate after experiments complete if you configure:

```bash
python runpod_deploy.py --models all --auto-terminate
```

### Manual Cleanup
```bash
# Terminate specific pod
python runpod_deploy.py --terminate <pod_id>

# Terminate all pods for this project
python runpod_deploy.py --cleanup-all
```

## Troubleshooting

### HuggingFace Authentication Failed
- Verify HF_TOKEN is set: `echo $HF_TOKEN`
- Check you've accepted model terms on HuggingFace
- Rebuild Docker image: `python runpod_deploy.py --build-only`

### Out of Memory (OOM)
- Reduce batch size: Edit `MODELS` dict in `runpod_deploy.py`
- Use larger GPU: `--gpu-type "NVIDIA A40"`
- Process fewer items: Reduce dataset size

### Pod Creation Failed
- Check RunPod balance
- Try different GPU type (some may be out of stock)
- Check API key: `echo $RUNPOD_API_KEY`

### Docker Push Failed
- Login to Docker Hub: `docker login`
- Check username: `echo $DOCKERHUB_USERNAME`
- Verify image built: `docker images | grep empathetic`

## Advanced Configuration

### Custom Batch Sizes
Edit `runpod_deploy.py`:

```python
MODELS = {
    'llama-3.1-8b': {'vram_gb': 20, 'batch_size': 2},  # Reduce if OOM
    'qwen2.5-7b': {'vram_gb': 18, 'batch_size': 4},
    'mistral-7b': {'vram_gb': 18, 'batch_size': 4},
    'llama-3-8b': {'vram_gb': 20, 'batch_size': 4},
}
```

### Custom Docker Image
```bash
# Build without pushing
docker build -f Dockerfile.gpu --build-arg HF_TOKEN=$HF_TOKEN -t myimage:latest .

# Use custom image
python runpod_deploy.py --image myimage:latest --skip-build
```

## Expected Timeline

- **Docker Build**: 5-10 minutes
- **Pod Startup**: 2-3 minutes per pod
- **Activation Extraction**: 30-45 minutes per model
- **SAE Training**: 20-30 minutes per model
- **Probe Training**: 10-15 minutes per model
- **Convergence Analysis**: 5 minutes

**Total (Parallel)**: 1-2 hours for all 4 models

## Next Steps

1. Deploy experiments: `python runpod_deploy.py --models all`
2. Monitor progress: https://www.runpod.io/console/pods
3. Download results when complete
4. Run local analysis: `python scripts/analyze_convergence.py`
5. Generate report: `python scripts/create_report.py`

## References

- RunPod Documentation: https://docs.runpod.io/
- RunPod API: https://docs.runpod.io/graphql-api/overview
- Docker Deployment: https://www.runpod.io/articles/guides/ai-workflows-with-docker-gpu-cloud
