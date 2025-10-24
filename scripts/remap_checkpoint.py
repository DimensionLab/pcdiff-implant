#!/usr/bin/env python3
"""
Quick patch script to fix checkpoint loading with reduced GPUs.
This loads the checkpoint, remaps all tensors to CPU, and saves it back.
Run this once before training with 7 GPUs.
"""

import torch
import sys
import argparse

def remap_checkpoint_to_cpu(checkpoint_path, output_path=None):
    """Load checkpoint and remap all tensors to CPU"""
    if output_path is None:
        output_path = checkpoint_path.replace('.pth', '_remapped.pth')
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    # Load with map_location='cpu' to avoid device issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Checkpoint contents: {list(checkpoint.keys())}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Save the remapped checkpoint
    print(f"Saving remapped checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    
    print("✓ Checkpoint successfully remapped to CPU!")
    print(f"\nUse this checkpoint in your training script:")
    print(f"  --model {output_path}")
    
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remap checkpoint tensors to CPU')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default=None, help='Output path (default: adds _remapped suffix)')
    
    args = parser.parse_args()
    
    remap_checkpoint_to_cpu(args.checkpoint, args.output)

