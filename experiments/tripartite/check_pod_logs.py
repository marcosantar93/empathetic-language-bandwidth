#!/usr/bin/env python3
"""Check if pods are actually running experiments or stuck."""
import os
import requests
import sys

RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
RUNPOD_API_BASE = "https://api.runpod.io/graphql"

def runpod_query(query, variables=None):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {RUNPOD_API_KEY}'
    }
    payload = {'query': query}
    if variables:
        payload['variables'] = variables
    response = requests.post(RUNPOD_API_BASE, json=payload, headers=headers)
    return response.json()

def check_pod_detailed(pod_id):
    """Get detailed pod information."""
    query = """
    query {
        pod(input: {podId: "%s"}) {
            id
            name
            desiredStatus
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
    """ % pod_id
    
    result = runpod_query(query)
    if 'data' in result and result['data']['pod']:
        pod = result['data']['pod']
        print(f"\nPod: {pod['name']} ({pod_id})")
        print(f"  Status: {pod.get('desiredStatus', 'UNKNOWN')}")
        
        runtime = pod.get('runtime') or {}
        uptime = runtime.get('uptimeInSeconds', 0)
        print(f"  Uptime: {uptime}s")
        
        gpus = runtime.get('gpus') or []
        if gpus:
            for gpu in gpus:
                util = gpu.get('gpuUtilPercent', 0)
                mem = gpu.get('memoryUtilPercent', 0)
                print(f"  GPU Util: {util}% | GPU Mem: {mem}%")
        else:
            print(f"  GPU Util: No data yet")
            
        return uptime, runtime.get('gpus')
    return 0, None

# Check our 4 pods
pod_ids = [
    '6gjgmmi8li4aq0',  # llama-3.1-8b
    '02126ik26px328',  # qwen2.5-7b
    'tu6saazp5mv3y1',  # mistral-7b
    'mlqee4qkgip0k6',  # llama-3-8b
]

print("="*70)
print("Detailed Pod Status Check")
print("="*70)

stuck_pods = []
for pod_id in pod_ids:
    uptime, gpus = check_pod_detailed(pod_id)
    
    # Check if pod might be stuck
    if uptime < 0:
        stuck_pods.append((pod_id, "negative uptime"))
    elif uptime == 0:
        stuck_pods.append((pod_id, "zero uptime after 5+ minutes"))

if stuck_pods:
    print("\n" + "="*70)
    print("⚠️  POTENTIAL ISSUES DETECTED")
    print("="*70)
    for pod_id, issue in stuck_pods:
        print(f"  {pod_id}: {issue}")
    print("\nRecommendation: Check RunPod web console for container logs")
else:
    print("\n✅ All pods appear to be running normally")

