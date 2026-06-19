from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def read_train_csv(path: str | Path, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    out = df[features].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if len(out) < 100:
        raise ValueError(f"Too few usable rows in {path}: {len(out)}")
    return out


def minmax_scale(values: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    min_vals = values.min(axis=0)
    max_vals = values.max(axis=0)
    scale = np.where((max_vals - min_vals) == 0, 1.0, max_vals - min_vals)
    scaled = (values - min_vals) / scale
    scaler = {
        "min": min_vals.tolist(),
        "max": max_vals.tolist(),
    }
    return scaled.astype(np.float32), scaler


def inverse_minmax_scale(values: np.ndarray, scaler: dict[str, list[float]]) -> np.ndarray:
    min_vals = np.asarray(scaler["min"], dtype=np.float64)
    max_vals = np.asarray(scaler["max"], dtype=np.float64)
    scale = np.where((max_vals - min_vals) == 0, 1.0, max_vals - min_vals)
    return values * scale + min_vals


def make_windows(values: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    windows = []
    for start in range(0, len(values) - seq_len + 1, stride):
        windows.append(values[start : start + seq_len])
    if not windows:
        raise ValueError(
            f"No windows created: len(values)={len(values)}, seq_len={seq_len}, stride={stride}"
        )
    return np.stack(windows).astype(np.float32)


def make_dataloader(windows: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    tensor = torch.tensor(windows, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def save_generated_csv(path: str | Path, values: np.ndarray, features: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(features)
        for row in values:
            writer.writerow([f"{float(x):.10g}" if math.isfinite(float(x)) else "" for x in row])


def ensure_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    dirs = {
        "root": root,
        "data": root / "data",
        "models": root / "models",
        "generated": root / "generated",
        "figures": root / "figures",
        "logs": root / "logs",
        "evaluation": root / "evaluation",
        "config": root / "config",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
