# Voxelization
To **train your own models** or **reproduce our results**, please follow the next steps.

## Install the Environment

### Modern Setup (Python 3.14 + uv) - RECOMMENDED
See the main [SETUP.md](../SETUP.md) guide for detailed installation instructions using Python 3.14 and `uv`.

**Quick start:**
```sh
# From project root
uv venv --python python3.14
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install -e .

# Install PyTorch3D (required)
uv pip install git+https://github.com/facebookresearch/pytorch3d.git@stable

# Install PyTorch Scatter (required)
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0%2Bcu130.html
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
We provide the scripts `preproc_skullbreak.py` and `preproc_skullfix.py` for preprocessing the required training data. Simply run:
```python
python voxelization/utils/preproc_skullbreak.py
python voxelization/utils/preproc_skullfix.py
```
To disable multiprocessing or change the number of threads use the flag `--multiprocessing` and `--threads`.
To split the training data into a smaller training and an evaluation set (e.g. to allow for early stopping), use the following commands:
```python
python voxelization/utils/split_skullbreak.py
python voxelization/utils/split_skullfix.py
```
## Train the Model
For training new models, we provide the script `train.py` and two exemplary commands on how to use it for the SkullBreak:
```python
python voxelization/train.py configs/train_skullbreak.yaml
```
and the SkullFix dataset:
```python
python voxelization/train.py configs/train_skullfix.yaml
```
The hyperparameters of the model can be adjusted in the corresponding config file. The hyperparameters we used to train our models are already set as default.
## Use the Model
For using the trained model, we provide the script `generate.py` and two exemplary commands on how to use it for the SkullBreak:
```python
python voxelization/generate.py voxelization/configs/gen_skullbreak.yaml
```
and the SkullFix dataset:
```python
python voxelization/generate.py voxelization/configs/gen_skullfix.yaml
```
For changing various parameters the two config files `gen_skullbreak.yaml` and `gen_skullfix.yaml` can be adjusted.
