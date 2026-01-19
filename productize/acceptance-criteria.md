# Acceptance Criteria

This document defines the acceptance criteria for PCDiff-Implant based on published benchmark results from the MICCAI 2023 paper.

## Evaluation Metrics

- **DSC (Dice Score)**: Measures volumetric overlap between predicted and ground truth implant (higher is better, ↑)
- **bDSC (10mm Boundary DSC)**: Dice score computed only on the 10mm boundary region (higher is better, ↑)
- **HD95 (95th Percentile Hausdorff Distance)**: Maximum surface distance at 95th percentile in mm (lower is better, ↓)

## SkullBreak Dataset Benchmark Results

Results from Table 1 of the PCDiff paper, evaluated on the SkullBreak test set (140 volumes):

| Model | DSC ↑ | bDSC ↑ | HD95 ↓ |
|-------|-------|--------|--------|
| 3D U-Net [1] | 0.87 | 0.91 | 2.32 |
| 3D U-Net (sparse) [2] | 0.71 | 0.80 | 4.60 |
| 2D U-Net [3] | 0.87 | 0.89 | 2.13 |
| **PCDiff (Ours)** | 0.86 | 0.88 | 2.51 |
| **PCDiff (Ours, n=5)** | 0.87 | 0.89 | 2.45 |

## Acceptance Thresholds

For production deployment, the PCDiff model should achieve scores that are competitive or similar with the published results:

| Metric | Minimum Threshold | Target |
|--------|-------------------|--------|
| DSC | ≥ 0.85 | ≥ 0.87 |
| bDSC | ≥ 0.87 | ≥ 0.89 |
| HD95 | ≤ 2.60 mm | ≤ 2.45 mm |

## References

1. Wodzinski, M., Daniol, M., Hemmerling, D.: **Improving the automatic cranial implant design in cranioplasty by linking different datasets.** In: Towards the Automatization of Cranial Implant Design in Cranioplasty II. pp. 29–44 (2021)

2. Kroviakov, A., Li, J., Egger, J.: **Sparse convolutional neural network for skull reconstruction.** In: Towards the Automatization of Cranial Implant Design in Cranioplasty II. pp. 80–94 (2021)

3. Yang, B., Fang, K., Li, X.: **Cranial implant prediction by learning an ensemble of slice-based skull completion networks.** In: Towards the Automatization of Cranial Implant Design in Cranioplasty II. pp. 95–104 (2021)
