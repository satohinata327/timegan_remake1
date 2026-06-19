#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch

from data_utils import (
    ensure_dirs,
    load_config,
    make_dataloader,
    make_windows,
    minmax_scale,
    read_train_csv,
    resolve_device,
    save_json,
    set_seed,
)
from timegan_model import (
    TimeGAN,
    TimeGANConfig,
    adversarial_loss,
    count_parameters,
    init_weights,
    loss_sqrt,
    moment_loss,
    random_generator,
    reconstruction_loss,
    supervised_loss,
)


def inverse_minmax_torch(
    values: torch.Tensor,
    min_vals: torch.Tensor,
    max_vals: torch.Tensor,
) -> torch.Tensor:
    scale = torch.where((max_vals - min_vals) == 0, torch.ones_like(max_vals), max_vals - min_vals)
    return values * scale + min_vals


def corr_flat(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)
    numerator = torch.mean(x_centered * y_centered)
    x_std = torch.sqrt(torch.mean(x_centered * x_centered) + eps)
    y_std = torch.sqrt(torch.mean(y_centered * y_centered) + eps)
    return numerator / (x_std * y_std + eps)


def corr_by_window(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_centered = x - torch.mean(x, dim=1, keepdim=True)
    y_centered = y - torch.mean(y, dim=1, keepdim=True)
    numerator = torch.mean(x_centered * y_centered, dim=1)
    x_std = torch.sqrt(torch.mean(x_centered * x_centered, dim=1) + eps)
    y_std = torch.sqrt(torch.mean(y_centered * y_centered, dim=1) + eps)
    return numerator / (x_std * y_std + eps)


def abs_autocorr_values(series: torch.Tensor, lags: list[int]) -> torch.Tensor:
    values = []
    abs_series = torch.abs(series)
    seq_len = abs_series.shape[1]
    for lag in lags:
        if lag >= seq_len:
            continue
        values.append(corr_flat(abs_series[:, :-lag], abs_series[:, lag:]))
    if not values:
        return torch.zeros(0, device=series.device)
    return torch.stack(values)


def abs_autocorr_loss(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    lags: list[int],
) -> torch.Tensor:
    losses = []
    for feature_idx in range(x_real.shape[2]):
        real_corr = abs_autocorr_values(x_real[:, :, feature_idx], lags).detach()
        fake_corr = abs_autocorr_values(x_fake[:, :, feature_idx], lags)
        if real_corr.numel() == 0:
            continue
        losses.append(torch.mean((fake_corr - real_corr) ** 2))
    if not losses:
        return torch.zeros((), device=x_real.device)
    return torch.mean(torch.stack(losses))


def cross_corr_loss(x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:
    real_corr = corr_flat(x_real[:, :, 0], x_real[:, :, 1]).detach()
    fake_corr = corr_flat(x_fake[:, :, 0], x_fake[:, :, 1])
    return (fake_corr - real_corr) ** 2


def rolling_corr_std_loss(x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:
    real_corrs = corr_by_window(x_real[:, :, 0], x_real[:, :, 1]).detach()
    fake_corrs = corr_by_window(x_fake[:, :, 0], x_fake[:, :, 1])
    real_std = torch.std(real_corrs, unbiased=True).detach()
    fake_std = torch.std(fake_corrs, unbiased=True)
    return (fake_std - real_std) ** 2


def train(config: dict) -> None:
    output_dir = config["output_dir"]
    dirs = ensure_dirs(output_dir)
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))

    features = list(config["features"])
    if len(features) != 2:
        raise ValueError("This remake script expects exactly two features: sp500 and DGS10")

    raw_df = read_train_csv(config["train_csv"], features)
    raw_values = raw_df.to_numpy()
    scaled, scaler = minmax_scale(raw_values)
    windows = make_windows(scaled, int(config["seq_len"]), int(config["stride"]))
    dataloader = make_dataloader(windows, int(config["batch_size"]), shuffle=True)

    min_vals = torch.tensor(scaler["min"], dtype=torch.float32, device=device).view(1, 1, -1)
    max_vals = torch.tensor(scaler["max"], dtype=torch.float32, device=device).view(1, 1, -1)
    abs_autocorr_lags = [int(lag) for lag in config.get("abs_autocorr_lags", [1, 5, 20])]
    abs_autocorr_weight = float(config.get("abs_autocorr_loss_weight", 1.0))
    cross_corr_weight = float(config.get("cross_corr_loss_weight", 1.0))
    rolling_corr_std_weight = float(config.get("rolling_corr_std_loss_weight", 1.0))

    save_json(dirs["data"] / "scaler.json", {"features": features, **scaler})
    save_json(
        dirs["data"] / "training_data_info.json",
        {
            "train_csv": config["train_csv"],
            "features": features,
            "n_rows": int(len(raw_df)),
            "seq_len": int(config["seq_len"]),
            "stride": int(config["stride"]),
            "n_windows": int(len(windows)),
            "added_losses": [
                "abs_autocorr_loss",
                "cross_corr_loss",
                "rolling_corr_std_loss",
            ],
            "abs_autocorr_lags": abs_autocorr_lags,
            "abs_autocorr_loss_weight": abs_autocorr_weight,
            "cross_corr_loss_weight": cross_corr_weight,
            "rolling_corr_std_loss_weight": rolling_corr_std_weight,
            "rolling_corr_std_training_note": (
                "Because seq_len=60, rolling_corr_std_loss uses the standard deviation "
                "of per-window 60-day correlations inside each mini-batch."
            ),
        },
    )

    model_config = TimeGANConfig(
        feature_dim=len(features),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        noise_dim=int(config["noise_dim"]),
        gamma=float(config["gamma"]),
    )
    model = TimeGAN(model_config).to(device)
    model.apply(init_weights)

    embedder_params = list(model.embedder.parameters()) + list(model.recovery.parameters())
    supervisor_params = list(model.supervisor.parameters())
    generator_params = list(model.generator.parameters()) + list(model.supervisor.parameters())
    discriminator_params = list(model.discriminator.parameters())

    e_optimizer = torch.optim.Adam(embedder_params, lr=float(config["learning_rate"]))
    s_optimizer = torch.optim.Adam(supervisor_params, lr=float(config["learning_rate"]))
    g_optimizer = torch.optim.Adam(generator_params, lr=float(config["learning_rate"]))
    d_optimizer = torch.optim.Adam(discriminator_params, lr=float(config["learning_rate"]))

    log_path = dirs["logs"] / "train_log.txt"
    start_time = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"device: {device}\n")
        log.write(f"parameters: {count_parameters(model)}\n")
        log.write(f"windows: {len(windows)}\n")
        log.write(f"config: {config}\n\n")

        def write(message: str) -> None:
            print(message)
            log.write(message + "\n")
            log.flush()

        iterator = iter(dataloader)

        def next_batch() -> torch.Tensor:
            nonlocal iterator
            try:
                (batch,) = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                (batch,) = next(iterator)
            return batch.to(device)

        write("Stage 1/3: embedder/recovery pretraining")
        for step in range(1, int(config["pretrain_iterations"]) + 1):
            x = next_batch()
            h = model.embed(x)
            x_tilde = model.recover(h)
            e_loss = reconstruction_loss(x, x_tilde)
            e_optimizer.zero_grad()
            e_loss.backward()
            e_optimizer.step()
            if step == 1 or step % 100 == 0:
                write(f"pretrain step={step} e_loss={float(e_loss.detach().cpu()):.6f}")

        write("Stage 2/3: supervisor pretraining")
        for step in range(1, int(config["supervised_iterations"]) + 1):
            x = next_batch()
            h = model.embed(x).detach()
            h_supervise = model.supervisor(h)
            s_loss = supervised_loss(h, h_supervise)
            s_optimizer.zero_grad()
            s_loss.backward()
            s_optimizer.step()
            if step == 1 or step % 100 == 0:
                write(f"supervised step={step} s_loss={float(s_loss.detach().cpu()):.6f}")

        write("Stage 3/3: joint adversarial training with abs autocorr and correlation losses")
        seq_len = int(config["seq_len"])
        noise_dim = int(config["noise_dim"])
        latest_abs_ac_loss = torch.zeros((), device=device)
        latest_cross_corr_loss = torch.zeros((), device=device)
        latest_roll_corr_std_loss = torch.zeros((), device=device)
        for step in range(1, int(config["joint_iterations"]) + 1):
            x = next_batch()
            batch_size = x.shape[0]

            for _ in range(2):
                z = random_generator(batch_size, seq_len, noise_dim, device)
                h = model.embed(x)
                h_supervise = model.supervisor(h)
                h_hat = model.generate_latent(z)
                x_hat = model.recover(h_hat)

                x_real_original = inverse_minmax_torch(x, min_vals, max_vals)
                x_fake_original = inverse_minmax_torch(x_hat, min_vals, max_vals)

                y_fake = model.discriminator(h_hat)
                g_loss_u = adversarial_loss(y_fake, True)
                g_loss_s = supervised_loss(h, h_supervise)
                g_loss_v = moment_loss(x, x_hat)
                latest_abs_ac_loss = abs_autocorr_loss(
                    x_real_original,
                    x_fake_original,
                    abs_autocorr_lags,
                )
                latest_cross_corr_loss = cross_corr_loss(x_real_original, x_fake_original)
                latest_roll_corr_std_loss = rolling_corr_std_loss(x_real_original, x_fake_original)
                g_loss = (
                    g_loss_u
                    + 100.0 * loss_sqrt(g_loss_s)
                    + 100.0 * g_loss_v
                    + abs_autocorr_weight * latest_abs_ac_loss
                    + cross_corr_weight * latest_cross_corr_loss
                    + rolling_corr_std_weight * latest_roll_corr_std_loss
                )

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                h = model.embed(x)
                x_tilde = model.recover(h)
                h_supervise = model.supervisor(h)
                e_loss_t0 = reconstruction_loss(x, x_tilde)
                e_loss_s = supervised_loss(h, h_supervise)
                e_loss = e_loss_t0 + 0.1 * e_loss_s
                e_optimizer.zero_grad()
                e_loss.backward()
                e_optimizer.step()

            z = random_generator(batch_size, seq_len, noise_dim, device)
            h_real = model.embed(x).detach()
            h_fake = model.generate_latent(z).detach()
            y_real = model.discriminator(h_real)
            y_fake = model.discriminator(h_fake)
            d_loss = adversarial_loss(y_real, True) + adversarial_loss(y_fake, False)
            if float(d_loss.detach().cpu()) > 0.15:
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            if step == 1 or step % 100 == 0:
                write(
                    "joint "
                    f"step={step} "
                    f"d_loss={float(d_loss.detach().cpu()):.6f} "
                    f"g_loss={float(g_loss.detach().cpu()):.6f} "
                    f"abs_ac_loss={float(latest_abs_ac_loss.detach().cpu()):.6f} "
                    f"cross_corr_loss={float(latest_cross_corr_loss.detach().cpu()):.6f} "
                    f"rolling_corr_std_loss={float(latest_roll_corr_std_loss.detach().cpu()):.6f} "
                    f"e_loss={float(e_loss.detach().cpu()):.6f}"
                )

        elapsed = time.time() - start_time
        write(f"training complete elapsed_sec={elapsed:.2f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "model_config": model_config.__dict__,
    }
    checkpoint_path = dirs["models"] / "timegan_abs_ac_corr.pt"
    torch.save(checkpoint, checkpoint_path)
    save_json(dirs["config"] / "used_config.json", config)
    print(f"Saved model to {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="timegan_remake1/config/timegan_seq60_abs_ac_corr.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
