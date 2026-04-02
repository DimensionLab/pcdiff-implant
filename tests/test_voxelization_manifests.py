from pathlib import Path

import numpy as np

from voxelization.src.manifests import load_manifest_records, parse_manifest_row


def test_parse_legacy_manifest_row(tmp_path: Path) -> None:
    base = tmp_path / "case001"
    record = parse_manifest_row([str(base)], tmp_path)

    assert record["format"] == "legacy"
    assert record["pointcloud_path"] == tmp_path / "case001_pc.npz"
    assert record["gt_psr_path"] == tmp_path / "case001_vox.npz"


def test_parse_generated_completion_manifest_row(tmp_path: Path) -> None:
    record = parse_manifest_row(
        ["samples.npy", "shift.npy", "scale.npy", "target_vox.npz", "2"],
        tmp_path,
    )

    assert record["format"] == "generated_completion"
    assert record["sample_points_path"] == tmp_path / "samples.npy"
    assert record["shift_path"] == tmp_path / "shift.npy"
    assert record["scale_path"] == tmp_path / "scale.npy"
    assert record["gt_psr_path"] == tmp_path / "target_vox.npz"
    assert record["ensemble_index"] == 2


def test_load_manifest_records_skips_comments_and_header(tmp_path: Path) -> None:
    manifest = tmp_path / "train.csv"
    manifest.write_text(
        "# comment\n"
        "sample_points_path,shift_path,scale_path,gt_psr_path,ensemble_index\n"
        "samples.npy,shift.npy,scale.npy,target_vox.npz,1\n",
        encoding="utf-8",
    )

    records = load_manifest_records(manifest)

    assert len(records) == 1
    assert records[0]["format"] == "generated_completion"
    assert records[0]["ensemble_index"] == 1
