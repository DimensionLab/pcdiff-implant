# Voxelization
To **train your own models** or **reproduce our results**, please follow the next steps.

## Install the Environment

### Modern Setup (Python 3.10 + uv) - RECOMMENDED
See the main [SETUP.md](../SETUP.md) guide for detailed installation instructions using Python 3.10 and `uv`.

**Quick start:**
```sh
# From project root
uv venv --python 3.10
source .venv/bin/activate
uv pip install "torch==2.5.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install "torchvision==0.20.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install -e .

# Install PyTorch3D (required, builds from source ~5-10 min)
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install PyTorch Scatter (required, pre-built wheel)
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

### Legacy Setup (Conda)
If you prefer the original conda environment (Python 3.8, PyTorch 1.12.0):
```sh
mamba env create -f voxelization/vox_env.yaml
mamba activate vox
mamba install pytorch3d -c pytorch3d
mamba install pytorch-scatter -c pyg
```

## Data preprocessing
We provide the scripts `preproc_skullbreak.py` and `preproc_skullfix.py` for preprocessing the required training data. 

**Note:** The dataset should be located at `pcdiff/datasets/SkullBreak/` (not `datasets/SkullBreak/`).

Run from the project root:
```bash
python voxelization/utils/preproc_skullbreak.py --root pcdiff/datasets/SkullBreak --csv pcdiff/datasets/SkullBreak/skullbreak.csv
python voxelization/utils/preproc_skullfix.py --root pcdiff/datasets/SkullFix --csv pcdiff/datasets/SkullFix/skullfix.csv
```
To disable multiprocessing or change the number of threads use the flag `--multiprocessing` and `--threads`.

To split the training data into a smaller training and an evaluation set (e.g. to allow for early stopping), use the following commands:
```bash
python voxelization/utils/split_skullbreak.py --root pcdiff/datasets/SkullBreak
python voxelization/utils/split_skullfix.py --root pcdiff/datasets/SkullFix
```
## Train the Model
For training new models, we provide the script `train.py` and two exemplary commands on how to use it.

**Option 1: Run from the `voxelization` directory (recommended):**
```bash
cd voxelization
python train.py configs/train_skullbreak.yaml  # SkullBreak dataset
python train.py configs/train_skullfix.yaml    # SkullFix dataset
```

**Option 2: Run from the project root:**
```bash
python voxelization/train.py voxelization/configs/train_skullbreak.yaml  # SkullBreak dataset
python voxelization/train.py voxelization/configs/train_skullfix.yaml    # SkullFix dataset
```

The hyperparameters of the model can be adjusted in the corresponding config file. The hyperparameters we used to train our models are already set as default.

### Weights & Biases Integration

The training script supports [Weights & Biases](https://wandb.ai/) for experiment tracking and visualization. By default, wandb will be used if it's installed.

**To use wandb:**
```bash
# Login to wandb (only needed once)
wandb login

# Train with wandb (default behavior)
cd voxelization
python train.py configs/train_skullbreak.yaml

# Customize wandb settings
python train.py configs/train_skullbreak.yaml \
  --wandb-project my-project \
  --wandb-entity my-team \
  --wandb-name my-experiment-name
```

**To disable wandb:**
```bash
python train.py configs/train_skullbreak.yaml --no-wandb
```

**Logged metrics include:**
- Training loss (overall and per-component)
- Validation metrics (PSR L2, IoU, etc.)
- Model parameters and gradients
- Best model checkpoints as artifacts
- Hyperparameters and system info

### Training Output

The training script provides professional, informative logging with:
- **Progress bars** showing real-time training progress per epoch
- **Epoch summaries** with average loss and timing information
- **Time tracking** with elapsed time, time per epoch, and ETA
- **Validation reports** with detailed metrics when validation runs
- **Visual indicators** for new best models and checkpoints

Example output:
```
================================================================================
                          TRAINING CONFIGURATION                              
================================================================================
  Dataset: SkullData
  Total epochs: 2600
  Starting epoch: 1
  Batch size: 2
  Learning rate: 0.0005
  Samples per epoch: 114
  Iterations per epoch: 57
  Validation every: 10 epochs
  Checkpoint every: 5 epochs
================================================================================

Epoch 001/2600: 100%|██████████| 57/57 [00:48<00:00,  1.18it/s, loss=0.0099, avg_loss=0.0156]

────────────────────────────────────────────────────────────────────────────────
Epoch 001/2600 Summary:
  Loss: 0.015623 (psr=0.015623)
  Time: 0:00:48 | Elapsed: 0:00:48 | ETA: 34:38:24
────────────────────────────────────────────────────────────────────────────────

Epoch 010/2600: 100%|██████████| 57/57 [00:47<00:00,  1.20it/s, loss=0.0088, avg_loss=0.0091]

════════════════════════════════════════════════════════════════════════════════
                                  VALIDATION                                    
════════════════════════════════════════════════════════════════════════════════
  Metric (psr_l2): 0.008234
  iou: 0.923456

  ⭐ NEW BEST MODEL ⭐
  Best psr_l2: 0.008234 (epoch 10)

════════════════════════════════════════════════════════════════════════════════

  💾 Checkpoint saved at epoch 10
```
## Use the Model
For using the trained model, we provide the script `generate.py` and two exemplary commands on how to use it.

**Option 1: Run from the `voxelization` directory (recommended):**
```bash
cd voxelization
python generate.py configs/gen_skullbreak.yaml  # SkullBreak dataset
python generate.py configs/gen_skullfix.yaml    # SkullFix dataset
```

**Option 2: Run from the project root:**
```bash
python voxelization/generate.py voxelization/configs/gen_skullbreak.yaml  # SkullBreak dataset
python voxelization/generate.py voxelization/configs/gen_skullfix.yaml    # SkullFix dataset
```

For changing various parameters the two config files `gen_skullbreak.yaml` and `gen_skullfix.yaml` can be adjusted.
