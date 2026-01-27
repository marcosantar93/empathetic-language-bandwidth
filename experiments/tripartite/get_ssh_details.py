#!/usr/bin/env python3
"""Get SSH connection details for all pods"""
import os
import json
import requests

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_API_URL = "https://api.runpod.io/graphql"

def graphql_query(query):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    response = requests.post(RUNPOD_API_URL, json={"query": query}, headers=headers)
    return response.json()

# Load pod info
with open("empathy_pods.json", 'r') as f:
    pods = json.load(f)

print("Getting SSH connection details...\n")

for pod in pods:
    pod_id = pod['id']
    model = pod['model']
    
    query = f'''
    query {{
        pod(input: {{podId: "{pod_id}"}}) {{
            id
            name
            runtime {{
                ports {{
                    privatePort
                    publicPort
                    type
                    ip
                }}
            }}
        }}
    }}
    '''
    
    result = graphql_query(query)
    pod_data = result.get('data', {}).get('pod', {})
    runtime = pod_data.get('runtime', {}) or {}
    ports = runtime.get('ports', [])
    
    print(f"Pod: {model} ({pod_id})")
    
    ssh_info = None
    for port in ports:
        if port.get('privatePort') == 22:
            ssh_info = {
                'ip': port.get('ip'),
                'port': port.get('publicPort')
            }
            print(f"  SSH: ssh root@{port.get('ip')} -p {port.get('publicPort')}")
            break
    
    if not ssh_info:
        print(f"  SSH: Not available yet")
    
    # Save for later
    pod['ssh'] = ssh_info
    print()

# Save updated pod info
with open("empathy_pods.json", 'w') as f:
    json.dump(pods, f, indent=2)

print("\nSSH info saved to empathy_pods.json")
print("\nRunPod uses SSH key authentication. Checking for keys...")
