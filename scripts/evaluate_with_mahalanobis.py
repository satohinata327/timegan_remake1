#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def copy_generated_as_mask(generated_dir: Path, mask_dir: Path) -> list[Path]:
    mask_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for idx, path in enumerate(sorted(generated_dir.glob("*.csv")), start=1):
        output_path = mask_dir / f"mask{idx}_timegan.csv"
        shutil.copy2(path, output_path)
        written.append(output_path)
    if not written:
        raise FileNotFoundError(f"No generated CSV files found in {generated_dir}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", default="timegan_remake1/runs/seq60_abs_ac/generated")
    parser.add_argument("--work-mask-dir", default="timegan_remake1/runs/seq60_abs_ac/evaluation/mahalanobis_input")
    parser.add_argument("--output-dir", default="timegan_remake1/runs/seq60_abs_ac/evaluation/mahalanobis_results")
    parser.add_argument("--mahalanobis-script", default="timegan_remake1/mahalanobis_eval/scripts/run_mahalanobis_eval.py")
    parser.add_argument("--train-csv", default="timegan_remake1/data/train_sp500_us10y.csv")
    args = parser.parse_args()

    generated_dir = Path(args.generated_dir)
    mask_dir = Path(args.work_mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = copy_generated_as_mask(generated_dir, mask_dir)
    print(f"Prepared {len(paths)} generated files for Mahalanobis evaluation")

    cmd = [
        "python3",
        args.mahalanobis_script,
        "--train-csv",
        args.train_csv,
        "--mask-dir",
        str(mask_dir),
        "--output-dir",
        str(output_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved evaluation results to {output_dir}")


if __name__ == "__main__":
    main()
