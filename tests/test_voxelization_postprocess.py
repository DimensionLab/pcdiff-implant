import numpy as np

from voxelization.src.postprocess import build_refined_complete_mask, select_implant_region


def test_bbox_postprocess_keeps_component_near_defect() -> None:
    implant = np.zeros((16, 16, 16), dtype=np.float32)
    implant[2:4, 2:4, 2:4] = 1
    implant[10:13, 10:13, 10:13] = 1
    defect_points = np.array([[11.0, 11.0, 11.0], [12.0, 12.0, 12.0]], dtype=np.float32)

    filtered = select_implant_region(defect_points, implant, method="bbox", bbox_margin_voxels=1)

    assert filtered.sum() == 27
    assert filtered[11, 11, 11] == 1
    assert filtered[2, 2, 2] == 0


def test_postprocess_falls_back_to_largest_component_when_region_misses() -> None:
    implant = np.zeros((16, 16, 16), dtype=np.float32)
    implant[1:5, 1:5, 1:5] = 1
    implant[12:14, 12:14, 12:14] = 1
    defect_points = np.array([[8.0, 8.0, 8.0]], dtype=np.float32)

    filtered = select_implant_region(defect_points, implant, method="bbox", bbox_margin_voxels=0)

    assert filtered.sum() == 64
    assert filtered[1, 1, 1] == 1
    assert filtered[12, 12, 12] == 0


def test_symmetry_refinement_fills_defect_region_from_mirrored_complete_prob() -> None:
    complete_prob = np.zeros((8, 8, 8), dtype=np.float32)
    complete_prob[5, 3:5, 3:5] = 1.0
    defect_points = np.array([[2.0, 3.0, 3.0], [2.0, 4.0, 4.0]], dtype=np.float32)

    refined = build_refined_complete_mask(
        complete_prob,
        defect_points,
        complete_prob.shape,
        threshold=0.3,
        symmetry_weight=1.0,
        symmetry_axis=0,
        symmetry_defect_only=True,
        defect_region_method="bbox",
        bbox_margin_voxels=0,
        radius_scale=1.0,
    )

    assert refined[2, 3, 3] == 1
    assert refined[2, 4, 4] == 1
    assert refined[5, 3, 3] == 1


def test_symmetry_refinement_preserves_non_defect_region_when_defect_only() -> None:
    complete_prob = np.zeros((8, 8, 8), dtype=np.float32)
    complete_prob[6, 2, 2] = 1.0
    defect_points = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

    refined = build_refined_complete_mask(
        complete_prob,
        defect_points,
        complete_prob.shape,
        threshold=0.2,
        symmetry_weight=1.0,
        symmetry_axis=0,
        symmetry_defect_only=True,
        defect_region_method="bbox",
        bbox_margin_voxels=0,
        radius_scale=1.0,
    )

    assert refined[1, 1, 1] == 0
    assert refined[6, 2, 2] == 1
