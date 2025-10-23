# Point Cloud Diffusion Model for Anatomy Restoration
To **train your own models** or **reproduce our results**, please follow the next steps.

## Install the Environment

### Modern Setup (Python 3.10 + uv) - RECOMMENDED
See the main [SETUP.md](../SETUP.md) guide for detailed installation instructions using Python 3.10 and `uv`.

**Quick start (single node, single GPU):**
```sh
# From project root
uv venv --python 3.10
source .venv/bin/activate
uv pip install "torch==2.5.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install "torchvision==0.20.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install -e .
```

**Multi-GPU (8x H100):**
```sh
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 64
```
The script automatically respects `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` from torchrun.

### Legacy Setup (Conda)
If you prefer the original conda environment (Python 3.6, PyTorch 1.7.1):
```sh
mamba env create -f pcdiff/pcd_env.yaml
mamba activate pcd
```
## Data preprocessing
Make sure you downloaded the data sets and stored them in the correct folder (see data structure).
We provide the scripts `preproc_skullbreak.py` and `preproc_skullfix.py` to preprocess the datasets.
Simply run the following commands:

```python
python pcdiff/utils/preproc_skullbreak.py
python pcdiff/utils/preproc_skullfix.py
```
To disable multiprocessing or change the number of threads use the flags `--multiprocessing` and `--threads`. 

Preprocessing the data may take some hours (depending on your hardware). We highly recommend to use multiprocessing.

## Train/Test Split
To randomly split the data into a training and a test set run:
```python
python pcdiff/utils/split_skullbreak.py
python pcdiff/utils/split_skullfix.py
```
The script creates a `train.csv` and `test.csv` file in the corresponding folder of the dataset, which can be used for the `--path` flag during training.
## Train the Model

### Single GPU Training
For training new models on a single GPU, use the script `train_completion.py`:

**SkullBreak:**
```bash
python pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 8
```

**SkullFix:**
```bash
python pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullFix/train.csv \
    --dataset SkullFix \
    --bs 8
```

### Multi-GPU Training (Distributed)

For distributed training across multiple GPUs using `torchrun`:

**SkullBreak on 8x GPU:**
```bash
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 64 \
    --lr 1.6e-3
```

**SkullFix on 8x GPU:**
```bash
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullFix/train.csv \
    --dataset SkullFix \
    --bs 64 \
    --lr 1.6e-3
```

**Important:** When scaling to N GPUs:
- Set `--bs` to `N × 8` (e.g., 64 for 8 GPUs) to maintain per-GPU batch size of 8
- Scale learning rate linearly: `--lr` = `N × 2e-4` (e.g., 1.6e-3 for 8 GPUs)
- See [distributed-training.md](./distributed-training.md) for detailed guidance

### Background Training (Persistent Sessions)

To keep training running even if SSH disconnects:

```bash
# Start a tmux session
tmux new -s training

# Run your training command
torchrun --nproc_per_node=8 pcdiff/train_completion.py [your args]

# Detach from session: Press Ctrl+B, then D
# The training continues in the background

# Reattach later
tmux attach -t training

# View logs in real-time
tail -f pcdiff/output/train_completion/*/output.log
```

Alternative using `screen`:
```bash
screen -S training
torchrun --nproc_per_node=8 pcdiff/train_completion.py [your args]
# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

### Hyperparameters
We provide many flags to change the hyperparameters of the model (details in the code). The hyperparameters used to generate the presented results are set as default.

## Use the Model
For using a trained model, we provide the script `test_completion.py` and two exemplary commands on how to use it for the SkullBreak:
```bash
python pcdiff/test_completion.py \
    --path pcdiff/datasets/SkullBreak/test.csv \
    --dataset SkullBreak \
    --model MODELPATH \
    --eval_path pcdiff/datasets/SkullBreak/results
```

For the SkullFix data set (if you want to use the proposed ensembling method, use the `--num_ens` flag to specify the number of different implants to be generated):
```bash
python pcdiff/test_completion.py \
    --path pcdiff/datasets/SkullFix/test.csv \
    --dataset SkullFix \
    --num_ens 5 \
    --model MODELPATH \
    --eval_path pcdiff/datasets/SkullFix/results
```

**Using DDIM sampling** (faster inference with fewer steps):
```bash
python pcdiff/test_completion.py \
    --path pcdiff/datasets/SkullBreak/test.csv \
    --dataset SkullBreak \
    --model MODELPATH \
    --eval_path ./inference_results \
    --sampling_method ddim \
    --sampling_steps 50
```

**Note:** The test script processes one sample at a time internally (batch_size=1) regardless of the `--bs` flag. Ensembling is controlled by `--num_ens`.
