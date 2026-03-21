import argparse
import csv
import glob
import random
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Split SkullFix dataset into train/eval')
parser.add_argument('--root', type=str, default='pcdiff/datasets/SkullFix', 
                    help='Root directory of the SkullFix dataset')
args = parser.parse_args()

root = Path(args.root)
data = []

with open(root / 'train.csv', 'r', newline='') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        data.append(row[0].split('complete')[0] + 'voxelization/' + row[0].split('skull/')[1].split('.nrrd')[0])

train = random.sample(data, 65)
test = [elem for elem in data if elem not in train]

# Create voxelization directory if it doesn't exist
(root / 'voxelization').mkdir(parents=True, exist_ok=True)

# Training set
with open(root / 'voxelization/train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(train)):
        datapoint = train[i]
        writer.writerow([datapoint])

# Test set
with open(root / 'voxelization/eval.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(test)):
        datapoint = test[i]
        writer.writerow([datapoint])

print("Successfully created training and evaluation split for SkullFix")
