#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_utils import (
    ensure_dirs,
    inverse_minmax_scale,
    load_config,
    resolve_device,
    save_generated_csv,
    set_seed,
)
from timegan_model import TimeGAN, TimeGANConfig, random_generator


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[TimeGAN, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model_config = TimeGANConfig(**checkpoint["model_config"])
    model = TimeGAN(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def generate_series(model: TimeGAN, config: dict, device: torch.device, length: int) -> np.ndarray:
    seq_len = int(config["seq_len"])
    noise_dim = int(config["noise_dim"])
    n_chunks = int(np.ceil(length / seq_len))
    chunks = []
    with torch.no_grad():
        for _ in range(n_chunks):
            z = random_generator(1, seq_len, noise_dim, device)
            x_hat = model.generate(z).detach().cpu().numpy()[0]
            chunks.append(x_hat)
    values = np.concatenate(chunks, axis=0)[:length]
    return np.clip(values, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="timegan_remake1/config/timegan_seq60_abs_ac.json")
    parser.add_argument("--checkpoint", default="timegan_remake1/runs/seq60_abs_ac/models/timegan_abs_autocorr.pt")
    parser.add_argument("--scaler", default="timegan_remake1/runs/seq60_abs_ac/data/scaler.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-generated", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
    args = parser.parse_args()

    file_config = load_config(args.config)
    output_dir = args.output_dir or file_config["output_dir"]
    dirs = ensure_dirs(output_dir)
    set_seed(int(file_config["seed"]))
    device = resolve_device(file_config.get("device", "auto"))

    model, train_config = load_model(args.checkpoint, device)
    scaler = load_config(args.scaler)
    features = list(scaler["features"])
    num_generated = args.num_generated or int(file_config["num_generated"])
    length = args.length or int(file_config["generated_length"])

    for idx in range(1, num_generated + 1):
        scaled = generate_series(model, train_config, device, length)
        original_scale = inverse_minmax_scale(scaled, scaler)
        output_path = dirs["generated"] / f"timegan_generated_{idx:03d}.csv"
        save_generated_csv(output_path, original_scale, features)
        print(output_path)


if __name__ == "__main__":
    main()
