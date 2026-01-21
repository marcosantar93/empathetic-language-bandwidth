#!/bin/bash
# Monitor experiments and auto-terminate instance when complete

INSTANCE_IP="18.212.122.191"
KEY_FILE="$HOME/.ssh/Ec2Tutorial.pem"
INSTANCE_ID="i-0eba493a01e559f35"
REGION="us-east-1"

echo "============================================"
echo "Monitoring Experiments"
echo "============================================"
echo "Instance: $INSTANCE_ID"
echo "Expected duration: 4-5 hours"
echo ""

# Monitor progress every 5 minutes
while true; do
    # Check if experiment log exists and shows completion
    COMPLETION_CHECK=$(ssh -i $KEY_FILE -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP \
        "grep -c 'All experiments complete' empathetic-language-bandwidth/experiments/tripartite/experiment_run.log 2>/dev/null" || echo "0")
    
    if [ "$COMPLETION_CHECK" -ge "1" ]; then
        echo ""
        echo "✓ Experiments completed!"
        break
    fi
    
    # Show recent progress
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking progress..."
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP \
        "tail -5 empathetic-language-bandwidth/experiments/tripartite/experiment_run.log 2>/dev/null" || echo "  (Log not yet available)"
    
    sleep 300  # Check every 5 minutes
done

# Download results
echo ""
echo "============================================"
echo "Downloading Results"
echo "============================================"

mkdir -p ./experiments/tripartite/results_from_ec2
scp -i $KEY_FILE -o StrictHostKeyChecking=no -r \
    ubuntu@$INSTANCE_IP:~/empathetic-language-bandwidth/experiments/tripartite/results/* \
    ./experiments/tripartite/results_from_ec2/ 2>/dev/null || echo "Warning: Some files may not exist yet"

scp -i $KEY_FILE -o StrictHostKeyChecking=no \
    ubuntu@$INSTANCE_IP:~/empathetic-language-bandwidth/experiments/tripartite/experiment_run.log \
    ./experiments/tripartite/ 2>/dev/null

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
