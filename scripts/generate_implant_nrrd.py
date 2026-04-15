"""Generate implant NRRD volumes from complete_skull - defective_skull."""
import argparse
from pathlib import Path
import nrrd
import numpy as np
from tqdm import tqdm

DEFECTS = ["bilateral", "frontoorbital", "parietotemporal", "random_1", "random_2"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/SkullBreak"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    complete_dir = root / "complete_skull"
    defective_dir = root / "defective_skull"
    implant_dir = root / "implant"

    cases = sorted(p.stem for p in complete_dir.glob("*.nrrd"))
    print(f"Found {len(cases)} cases, {len(DEFECTS)} defect types each")

    generated = 0
    skipped = 0
    failed = 0

    for case_id in tqdm(cases):
        complete_path = complete_dir / f"{case_id}.nrrd"
        complete_vol, complete_hdr = nrrd.read(str(complete_path))

        for defect in DEFECTS:
            out_path = implant_dir / defect / f"{case_id}.nrrd"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            defective_path = defective_dir / defect / f"{case_id}.nrrd"
            if not defective_path.exists():
                print(f"  SKIP missing defective: {defective_path}")
                failed += 1
                continue

            try:
                defective_vol, _ = nrrd.read(str(defective_path))
                implant_vol = np.clip(complete_vol.astype(np.float32) - defective_vol.astype(np.float32), 0, None)
                implant_vol = (implant_vol > 0.5).astype(np.uint8)

                out_path.parent.mkdir(parents=True, exist_ok=True)
                nrrd.write(str(out_path), implant_vol, header=complete_hdr)
                generated += 1
            except Exception as e:
                print(f"  FAIL {case_id}/{defect}: {e}")
                failed += 1

    print(f"\nDone: generated={generated}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
