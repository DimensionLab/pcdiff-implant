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
# --commit-logs ensures every experiment log/artifact is committed in git.
python run_experiments.py --time-budget 900 --max-experiments 50 --commit-logs
```

## Files

| File | Purpose | Agent-editable? |
|------|---------|----------------|
| `prepare_pcdiff.py` | Data loading, Chamfer Distance eval, logging | No |
| `train_pcdiff.py` | Self-contained PCDiff training | **Yes** |
| `program_pcdiff.md` | Research directions for the LLM agent | Human-editable |
| `run_experiments.py` | Experiment orchestrator (LLM loop) | No |
| `run_manual_variant_campaign.py` | Fixed variant family runner (no LLM proposal step) | No |
| `run_voxelization_variant_campaign.py` | Voxelization variant campaign harness | No |
| `check_voxelization_env.py` | Preflight dependency checker for voxelization runs | No |
| `setup_runpod.py` | RunPod pod creation and management | No |

## Audit Logs

Each experiment now writes full audit artifacts under:

- `autoresearch/results/audit/exp_XXXX_<timestamp>/train_before.py`
- `autoresearch/results/audit/exp_XXXX_<timestamp>/train_after.py`
- `autoresearch/results/audit/exp_XXXX_<timestamp>/proposal.diff`
- `autoresearch/results/audit/exp_XXXX_<timestamp>/stdout.log`
- `autoresearch/results/audit/exp_XXXX_<timestamp>/stderr.log`
- `autoresearch/results/audit/exp_XXXX_<timestamp>/result.json`

To keep these in repository history, run campaigns with `--commit-logs`.

## Manual Campaign Runner

For pre-approved fixed-override families (e.g., attention/dropout ablations), use:

```bash
cd /workspace/pcdiff-implant/autoresearch
python run_manual_variant_campaign.py --campaign attn_dropout_ablation_long --time-budget 5400 --commit-logs
```

Dry run:

```bash
python run_manual_variant_campaign.py --campaign attn_dropout_ablation_long --dry-run
```

## Voxelization Campaign Runner

Run a SkullBreak voxelization ablation batch:

```bash
cd /workspace/pcdiff-implant/autoresearch
python ../voxelization/utils/preproc_skullbreak.py --root ../pcdiff/datasets/SkullBreak --csv ../pcdiff/datasets/SkullBreak/skullbreak.csv
python ../voxelization/utils/split_skullbreak.py --root ../pcdiff/datasets/SkullBreak
python run_voxelization_variant_campaign.py --campaign vox_skullbreak_ablation_v1 --epochs 80 --dry-run
```

Run actual training variants:

```bash
python run_voxelization_variant_campaign.py --campaign vox_skullbreak_ablation_v1 --epochs 80
```

Preflight environment check (recommended first):

```bash
python check_voxelization_env.py
```

## How It Works

1. LLM reads `program_pcdiff.md` and experiment history
2. Proposes a modification to `train_pcdiff.py`
3. Training runs for 15 min (fixed budget)
4. Evaluates Chamfer Distance on 10 validation cases (DDIM-50)
5. Accept if metric improves, reject and revert otherwise
6. Repeat
