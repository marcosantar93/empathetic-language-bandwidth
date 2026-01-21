#!/bin/bash
# Start experiments on all 4 pods via Jupyter terminal
# You can run this script OR copy-paste the commands manually

cat << 'EOF'
============================================================
INSTRUCTIONS: Start Experiments on All Pods
============================================================

1. Go to: https://www.runpod.io/console/pods
2. For EACH pod, click "Connect" → "Start Jupyter Lab"
3. Wait 30-60 seconds for Jupyter to load
4. Open a "Terminal" in Jupyter
5. Copy-paste these commands:

------------------------------------------------------------
POD 1: llama-3.1-8b (hj8fo0awl242v5)
------------------------------------------------------------
pip install -q huggingface_hub[cli] && \
huggingface-cli login --token $HF_TOKEN && \
cd /workspace && \
git clone https://github.com/marcosantar93/empathetic-language-bandwidth.git && \
cd empathetic-language-bandwidth && \
pip install -q --no-cache-dir -r requirements-gpu.txt && \
nohup python experiments/tripartite/scripts/run_all_gpu.py --models llama-3.1-8b --batch-size 2 > experiment.log 2>&1 &

# Monitor: tail -f experiment.log

------------------------------------------------------------
POD 2: qwen2.5-7b (8sb2lk9lx9qw7c)
------------------------------------------------------------
pip install -q huggingface_hub[cli] && \
huggingface-cli login --token $HF_TOKEN && \
cd /workspace && \
git clone https://github.com/marcosantar93/empathetic-language-bandwidth.git && \
cd empathetic-language-bandwidth && \
pip install -q --no-cache-dir -r requirements-gpu.txt && \
nohup python experiments/tripartite/scripts/run_all_gpu.py --models qwen2.5-7b --batch-size 2 > experiment.log 2>&1 &

# Monitor: tail -f experiment.log

------------------------------------------------------------
POD 3: mistral-7b (35s56r292qhxj5)
------------------------------------------------------------
pip install -q huggingface_hub[cli] && \
huggingface-cli login --token $HF_TOKEN && \
cd /workspace && \
git clone https://github.com/marcosantar93/empathetic-language-bandwidth.git && \
cd empathetic-language-bandwidth && \
pip install -q --no-cache-dir -r requirements-gpu.txt && \
nohup python experiments/tripartite/scripts/run_all_gpu.py --models mistral-7b --batch-size 2 > experiment.log 2>&1 &

# Monitor: tail -f experiment.log

------------------------------------------------------------
POD 4: llama-3-8b (397iqxfwztrv8m)
------------------------------------------------------------
pip install -q huggingface_hub[cli] && \
huggingface-cli login --token $HF_TOKEN && \
cd /workspace && \
git clone https://github.com/marcosantar93/empathetic-language-bandwidth.git && \
cd empathetic-language-bandwidth && \
pip install -q --no-cache-dir -r requirements-gpu.txt && \
nohup python experiments/tripartite/scripts/run_all_gpu.py --models llama-3-8b --batch-size 2 > experiment.log 2>&1 &

# Monitor: tail -f experiment.log

============================================================
TIPS:
============================================================
- Use "nohup ... &" so experiments continue if you close browser
- Monitor with: tail -f experiment.log
- Each experiment takes ~1-2 hours
- Results in: experiments/tripartite/results/
- GPU usage should show 80-100% when running

============================================================
To terminate all pods when done:
============================================================
python runpod_parallel.py --terminate-all

EOF
