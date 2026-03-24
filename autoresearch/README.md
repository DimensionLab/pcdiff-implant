# Autoresearch for PCDiff

Autonomous AI-driven ML research framework adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for improving cranial implant point cloud diffusion models.

## Quick Start

```bash
# On RunPod pod (with GPU):
cd /workspace/pcdiff-implant/autoresearch

# 1. Verify data access
python prepare_pcdiff.py

# 2. Run baseline
python train_pcdiff.py --baseline

# 3. Run autoresearch loop (requires OPENROUTER_API_KEY)
python run_experiments.py --time-budget 900 --max-experiments 50
```

## Files

| File | Purpose | Agent-editable? |
|------|---------|----------------|
| `prepare_pcdiff.py` | Data loading, Chamfer Distance eval, logging | No |
| `train_pcdiff.py` | Self-contained PCDiff training | **Yes** |
| `program_pcdiff.md` | Research directions for the LLM agent | Human-editable |
| `run_experiments.py` | Experiment orchestrator (LLM loop) | No |
| `setup_runpod.py` | RunPod pod creation and management | No |

## How It Works

1. LLM reads `program_pcdiff.md` and experiment history
2. Proposes a modification to `train_pcdiff.py`
3. Training runs for 15 min (fixed budget)
4. Evaluates Chamfer Distance on 10 validation cases (DDIM-50)
5. Accept if metric improves, reject and revert otherwise
6. Repeat
