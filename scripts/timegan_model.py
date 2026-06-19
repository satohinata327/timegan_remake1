from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TimeGANConfig:
    feature_dim: int
    hidden_dim: int
    num_layers: int
    noise_dim: int
    gamma: float = 1.0


class RecurrentBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return self.activation(self.proj(h))


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return self.proj(h)


class TimeGAN(nn.Module):
    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.config = config
        self.embedder = RecurrentBlock(
            config.feature_dim,
            config.hidden_dim,
            config.num_layers,
            config.hidden_dim,
        )
        self.recovery = RecurrentBlock(
            config.hidden_dim,
            config.hidden_dim,
            config.num_layers,
            config.feature_dim,
        )
        self.generator = RecurrentBlock(
            config.noise_dim,
            config.hidden_dim,
            config.num_layers,
            config.hidden_dim,
        )
        self.supervisor = RecurrentBlock(
            config.hidden_dim,
            config.hidden_dim,
            max(config.num_layers - 1, 1),
            config.hidden_dim,
        )
        self.discriminator = DiscriminatorBlock(
            config.hidden_dim,
            config.hidden_dim,
            config.num_layers,
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedder(x)

    def recover(self, h: torch.Tensor) -> torch.Tensor:
        return self.recovery(h)

    def generate_latent(self, z: torch.Tensor) -> torch.Tensor:
        e_hat = self.generator(z)
        return self.supervisor(e_hat)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        h_hat = self.generate_latent(z)
        return self.recover(h_hat)


def reconstruction_loss(x: torch.Tensor, x_tilde: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.sqrt(nn.functional.mse_loss(x_tilde, x) + 1e-8)


def supervised_loss(h: torch.Tensor, h_supervise: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(h[:, 1:, :], h_supervise[:, :-1, :])


def adversarial_loss(logits: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    target = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
    return nn.functional.binary_cross_entropy_with_logits(logits, target)


def moment_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    mean_loss = torch.mean(torch.abs(torch.mean(x_hat, dim=0) - torch.mean(x, dim=0)))
    std_loss = torch.mean(
        torch.abs(
            torch.sqrt(torch.var(x_hat, dim=0) + 1e-6)
            - torch.sqrt(torch.var(x, dim=0) + 1e-6)
        )
    )
    return mean_loss + std_loss


def random_generator(batch_size: int, seq_len: int, noise_dim: int, device: torch.device) -> torch.Tensor:
    return torch.rand(batch_size, seq_len, noise_dim, device=device)


def loss_sqrt(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(value + 1e-8)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def safe_sqrt(value: float) -> float:
    return math.sqrt(max(value, 0.0))
