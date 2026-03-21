#!/usr/bin/env python3
"""
Test client for PCDiff Runpod Serverless endpoint.

Usage:
    python test_client.py --endpoint YOUR_ENDPOINT_ID --api_key YOUR_API_KEY --input defective_skull.npy

Environment Variables (alternative to command line):
    RUNPOD_API_KEY: Your Runpod API key
    RUNPOD_ENDPOINT_ID: Your endpoint ID
"""

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import requests


def encode_numpy_file(file_path):
    """Encode a numpy file to base64."""
    data = np.load(file_path)
    buffer = io.BytesIO()
    np.save(buffer, data)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def submit_job(endpoint_id, api_key, input_data, input_format='base64', 
               num_ensemble=1, sampling_steps=1000, output_prefix=None):
    """Submit a job to the Runpod endpoint."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "defective_skull": input_data,
            "input_format": input_format,
            "num_ensemble": num_ensemble,
            "sampling_steps": sampling_steps,
        }
    }
    
    if output_prefix:
        payload["input"]["output_prefix"] = output_prefix
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    return response.json()


def check_status(endpoint_id, api_key, job_id):
    """Check the status of a job."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()


def wait_for_completion(endpoint_id, api_key, job_id, timeout=600, poll_interval=5):
    """Wait for a job to complete."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        result = check_status(endpoint_id, api_key, job_id)
        status = result.get('status')
        
        print(f"  Status: {status}")
        
        if status == 'COMPLETED':
            return result
        elif status == 'FAILED':
            raise Exception(f"Job failed: {result.get('error', 'Unknown error')}")
        elif status in ['IN_QUEUE', 'IN_PROGRESS']:
            time.sleep(poll_interval)
        else:
            raise Exception(f"Unknown status: {status}")
    
    raise TimeoutError(f"Job did not complete within {timeout} seconds")


def main():
    parser = argparse.ArgumentParser(description='Test PCDiff Runpod Serverless endpoint')
    parser.add_argument('--endpoint', type=str, 
                        default=os.environ.get('RUNPOD_ENDPOINT_ID'),
                        help='Runpod endpoint ID')
    parser.add_argument('--api_key', type=str,
                        default=os.environ.get('RUNPOD_API_KEY'),
                        help='Runpod API key')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input .npy file (defective skull point cloud)')
    parser.add_argument('--num_ensemble', type=int, default=1,
                        help='Number of ensemble samples')
    parser.add_argument('--sampling_steps', type=int, default=1000,
                        help='Diffusion sampling steps')
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='Prefix for S3 output keys')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Timeout in seconds')
    parser.add_argument('--async_mode', action='store_true',
                        help='Submit job and exit without waiting')
    
    args = parser.parse_args()
    
    if not args.endpoint:
        raise ValueError("Endpoint ID required. Set RUNPOD_ENDPOINT_ID or use --endpoint")
    if not args.api_key:
        raise ValueError("API key required. Set RUNPOD_API_KEY or use --api_key")
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print("=" * 60)
    print("  PCDiff Serverless Test Client")
    print("=" * 60)
    print(f"Endpoint: {args.endpoint}")
    print(f"Input: {input_path}")
    print(f"Ensemble: {args.num_ensemble}")
    print(f"Sampling steps: {args.sampling_steps}")
    print()
    
    # Encode input
    print("Encoding input file...")
    encoded_input = encode_numpy_file(input_path)
    print(f"  Encoded size: {len(encoded_input) / 1024:.1f} KB")
    
    # Submit job
    print("\nSubmitting job...")
    result = submit_job(
        args.endpoint, 
        args.api_key, 
        encoded_input,
        num_ensemble=args.num_ensemble,
        sampling_steps=args.sampling_steps,
        output_prefix=args.output_prefix
    )
    
    job_id = result.get('id')
    print(f"  Job ID: {job_id}")
    
    if args.async_mode:
        print("\nAsync mode - job submitted. Check status with:")
        print(f"  curl 'https://api.runpod.ai/v2/{args.endpoint}/status/{job_id}' \\")
        print(f"       -H 'Authorization: Bearer {args.api_key[:10]}...'")
        return
    
    # Wait for completion
    print("\nWaiting for completion...")
    try:
        result = wait_for_completion(args.endpoint, args.api_key, job_id, 
                                     timeout=args.timeout)
    except TimeoutError as e:
        print(f"\n⚠ {e}")
        print("Job is still running. Check status manually.")
        return
    
    # Print results
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    
    output = result.get('output', {})
    
    if output.get('status') == 'success':
        print("\n✓ Inference completed successfully!")
        print("\nGenerated files:")
        for key, url in output.get('results', {}).items():
            print(f"  {key}: {url}")
        
        print("\nMetadata:")
        for key, value in output.get('metadata', {}).items():
            print(f"  {key}: {value}")
    else:
        print("\n✗ Inference failed!")
        print(f"Error: {output.get('error', 'Unknown error')}")
        if output.get('traceback'):
            print(f"\nTraceback:\n{output['traceback']}")


if __name__ == '__main__':
    main()

