#!/bin/bash
set -e

# Tripartite Empathy Decomposition - EC2 Deployment Script
# Launches instance, runs experiments, downloads results, terminates instance

INSTANCE_TYPE="g5.xlarge"
AMI_ID="ami-0c7217cdde317cfec"  # Deep Learning AMI (Ubuntu 20.04) us-east-1
KEY_NAME="Ec2tutorial"
KEY_FILE="$HOME/.ssh/Ec2Tutorial.pem"
SECURITY_GROUP="sg-0b6bbcac4567fcd67"  # docker-builder-sg
REGION="us-east-1"
REPO_URL="https://github.com/marcosantar93/empathetic-language-bandwidth.git"

echo "============================================"
echo "EC2 Tripartite Empathy Deployment"
echo "============================================"
echo "Instance type: $INSTANCE_TYPE"
echo "Region: $REGION"
echo ""

# Step 1: Launch EC2 instance
echo "Step 1: Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --region $REGION \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=tripartite-empathy},{Key=AutoTerminate,Value=true}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance ID: $INSTANCE_ID"

# Step 2: Wait for instance to be running
echo ""
echo "Step 2: Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
INSTANCE_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance IP: $INSTANCE_IP"

# Step 3: Wait for SSH to be available
echo ""
echo "Step 3: Waiting for SSH to be available..."
sleep 30
for i in {1..30}; do
    if ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$INSTANCE_IP "echo 'SSH ready'" 2>/dev/null; then
        echo "SSH connection established"
        break
    fi
    echo "  Attempt $i/30..."
    sleep 10
done

# Step 4: Set up and run experiments
echo ""
echo "Step 4: Setting up experiments on instance..."

ssh -i $KEY_FILE -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP << 'ENDSSH'
set -e

echo "=== Cloning repository ==="
git clone https://github.com/marcosantar93/empathetic-language-bandwidth.git
cd empathetic-language-bandwidth

echo ""
echo "=== Building Docker image ==="
docker build -f Dockerfile.gpu -t tripartite-empathy .

echo ""
echo "=== Running experiments ==="
docker run --gpus all \
    -v $(pwd)/experiments/tripartite:/app/experiments/tripartite \
    tripartite-empathy \
    python experiments/tripartite/scripts/run_all_gpu.py --models all

echo ""
echo "=== Experiments complete! ==="
ls -lh experiments/tripartite/results/

ENDSSH

# Step 5: Download results
echo ""
echo "Step 5: Downloading results..."
mkdir -p ./experiments/tripartite/results_from_ec2
scp -i $KEY_FILE -o StrictHostKeyChecking=no -r ubuntu@$INSTANCE_IP:~/empathetic-language-bandwidth/experiments/tripartite/results/* \
    ./experiments/tripartite/results_from_ec2/

echo ""
echo "Results downloaded to: ./experiments/tripartite/results_from_ec2/"

# Step 6: Terminate instance
echo ""
echo "Step 6: Terminating instance..."
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION

echo ""
echo "============================================"
echo "DEPLOYMENT COMPLETE!"
echo "============================================"
echo "Instance $INSTANCE_ID terminated"
echo "Results available in: ./experiments/tripartite/results_from_ec2/"
echo ""
echo "Next steps:"
echo "  1. Review results in results_from_ec2/"
echo "  2. Commit results: git add experiments/tripartite/results_from_ec2/"
echo "  3. git commit -m 'Add tripartite decomposition results'"
echo "============================================"
