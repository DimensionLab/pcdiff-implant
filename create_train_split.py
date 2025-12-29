#!/usr/bin/env python3
"""Create train/test split for SkullBreak dataset."""

import os
from pathlib import Path
import random
import csv

def main():
    dataset_root = Path("datasets/SkullBreak")
    defective_dir = dataset_root / "defective_skull"
    implant_dir = dataset_root / "implant"
    
    # Collect all samples
    samples = []
    defect_types = ["bilateral", "frontoorbital", "parietotemporal", "random_1", "random_2"]
    
    for defect_type in defect_types:
        defective_path = defective_dir / defect_type
        implant_path = implant_dir / defect_type
        
        if not defective_path.exists():
            print(f"Warning: {defective_path} not found")
            continue
            
        for npy_file in defective_path.glob("*.npy"):
            sample_name = npy_file.stem  # e.g., "024_surf"
            implant_file = implant_path / npy_file.name
            
            if implant_file.exists():
                samples.append({
                    "defective": str(npy_file),
                    "implant": str(implant_file),
                    "defect_type": defect_type,
                    "name": sample_name
                })
            else:
                print(f"Warning: No implant for {npy_file}")
    
    print(f"Found {len(samples)} valid samples")
    
    # Shuffle and split (80% train, 20% test)
    random.seed(42)
    random.shuffle(samples)
    
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")
    
    # Write CSV files
    def write_csv(filepath, samples):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['defective', 'implant', 'defect_type', 'name'])
            for s in samples:
                writer.writerow([s['defective'], s['implant'], s['defect_type'], s['name']])
    
    write_csv(dataset_root / "train.csv", train_samples)
    write_csv(dataset_root / "test.csv", test_samples)
    
    print(f"Created {dataset_root}/train.csv and {dataset_root}/test.csv")

if __name__ == "__main__":
    main()

