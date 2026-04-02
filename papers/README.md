# Recovered Research Packet

Date: 2026-03-22

This repository did not include the referenced `papers/` folder, so this file serves as the reconstructed research packet for the current workspace.

## Primary Sources

- MICCAI 2023 paper: "Point Cloud Diffusion Models for Automatic Implant Generation"
  - Project page: https://pfriedri.github.io/pcdiff-implant-io/
  - arXiv: https://arxiv.org/abs/2303.08061
  - DOI landing page: https://doi.org/10.1007/978-3-031-43996-4_11
- Dataset reference for SkullBreak / SkullFix:
  - Data article landing page referenced by the project page and README: https://www.sciencedirect.com/science/article/pii/S2352340921001864
- Evaluation metrics reference cited by the README:
  - https://github.com/OldaKodym/evaluation_metrics

## What This Packet Establishes

- The public claim source is the MICCAI 2023 paper plus the project README tables.
- The public datasets named by the repo are SkullBreak and SkullFix.
- The metric implementation used in this repository matches the external `evaluation_metrics` reference for DSC, 10 mm border DSC, and HD95.
- The executable benchmark definition for this workspace is documented in `benchmarking/benchmark_spec_2026-03-22.md`.

## Current Limits

- This packet restores source provenance and metric lineage, not the missing raw dataset artifacts.
- The repository still does not contain the full SkullBreak / SkullFix volumes needed to rerun the benchmark.
- The checked-in workspace still requires a local environment restore and local split regeneration before a reproduction claim is credible.

## Recommended Use

- Treat this folder as the source index for external papers and benchmark provenance.
- Treat `benchmarking/baseline_reproduction_audit_2026-03-22.md` as the reproducibility audit.
- Treat `benchmarking/benchmark_spec_2026-03-22.md` as the source-of-truth benchmark definition for the current codebase.
