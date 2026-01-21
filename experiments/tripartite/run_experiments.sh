#!/bin/bash
# Run experiments on EC2 and download results
set -e

INSTANCE_IP="18.212.122.191"
KEY_FILE="$HOME/.ssh/Ec2Tutorial.pem"
INSTANCE_ID="i-0eba493a01e559f35"
REGION="us-east-1"

echo "============================================"
echo "Running Tripartite Empathy Experiments"
echo "============================================"
echo "Instance: $INSTANCE_ID"
echo "IP: $INSTANCE_IP"
echo ""

# Wait for Docker build to complete
echo "Waiting for Docker build to complete..."
while true; do
    BUILD_STATUS=$(ssh -i $KEY_FILE -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP \
        "sudo docker images | grep tripartite-empathy | wc -l" 2>/dev/null || echo "0")

    if [ "$BUILD_STATUS" -ge "1" ]; then
        echo "✓ Docker image ready"
        break
    fi
    echo "  Still building..."
    sleep 30
done

# Run experiments
echo ""
echo "============================================"
echo "Starting Experiments (4-5 hours)"
echo "============================================"

ssh -i $KEY_FILE -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP << 'ENDSSH'
cd empathetic-language-bandwidth

echo "=== Running experiments ==="
echo "Started at: $(date)"
echo ""

sudo docker run --gpus all \
    -v $(pwd)/experiments/tripartite:/app/experiments/tripartite \
    tripartite-empathy \
    python experiments/tripartite/scripts/run_all_gpu.py --models all

echo ""
echo "=== Experiments complete at: $(date) ==="
ls -lh experiments/tripartite/results/

ENDSSH

# Download results
echo ""
echo "============================================"
echo "Downloading Results"
echo "============================================"

mkdir -p ./experiments/tripartite/results_from_ec2
scp -i $KEY_FILE -o StrictHostKeyChecking=no -r \
    ubuntu@$INSTANCE_IP:~/empathetic-language-bandwidth/experiments/tripartite/results/* \
    ./experiments/tripartite/results_from_ec2/

echo ""
echo "✓ Results downloaded to: ./experiments/tripartite/results_from_ec2/"

# Terminate instance
echo ""
echo "============================================"
echo "Terminating Instance"
echo "============================================"

aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION

echo ""
echo "✓ Instance $INSTANCE_ID terminated"
echo ""
echo "============================================"
echo "COMPLETE!"
echo "============================================"
echo "Results: ./experiments/tripartite/results_from_ec2/"
echo ""
echo "Next steps:"
echo "  1. Review convergence_report.json"
echo "  2. Check convergence_visualization.png"
echo "  3. Commit results to git"
echo "============================================"
