"""Lightweight projector heads for intermediate-layer alignment."""

from __future__ import annotations

from typing import Any, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp

XAVIER_UNIFORM = nn.initializers.xavier_uniform()
ZERO_INIT = nn.initializers.zeros
RESIDUAL_BRANCH_INIT = nn.initializers.normal(stddev=1e-4)

__all__ = [
    "ProjectorBase",
    "ResidualBottleneckMLP",
    "DeepResidualMLP",
    "MiniConvNeXtProjector",
    "build_projector",
    "count_parameters",
]


def _activation(x: jnp.ndarray, name: str) -> jnp.ndarray:
    if name == "silu":
        return nn.silu(x)
    if name == "gelu":
        return nn.gelu(x, approximate=True)
    raise ValueError(f"Unsupported projector activation '{name}'")


def count_parameters(params: Mapping[str, Any] | Any) -> int:
    """Count scalar parameters in a Flax parameter pytree."""
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


class ProjectorBase(nn.Module):
    """Common interface for LayerSync-style token projectors."""

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        raise NotImplementedError


class ResidualBottleneckMLP(ProjectorBase):
    """Default LayerSync projector: stable, cheap, and identity-like at init."""

    output_dim: int
    hidden_dim: int = 384
    depth: int = 4
    activation: str = "silu"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        if self.depth < 4 or self.depth > 6:
            raise ValueError(f"ResidualBottleneckMLP depth must be in [4, 6], got {self.depth}")

        residual = _project_residual(x, self.output_dim)
        y = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=True)(x)
        for _ in range(self.depth - 1):
            y = nn.Dense(
                self.hidden_dim,
                kernel_init=XAVIER_UNIFORM,
                bias_init=ZERO_INIT,
            )(y)
            y = _activation(y, self.activation)
        if self.dropout_rate > 0.0:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        y = nn.Dense(
            self.output_dim,
            kernel_init=RESIDUAL_BRANCH_INIT,
            bias_init=ZERO_INIT,
        )(y)
        return residual + y


class _ResidualMLPBlock(nn.Module):
    output_dim: int
    hidden_dim: int
    activation: str
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = _project_residual(x, self.output_dim)
        y = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=True)(x)
        y = nn.Dense(
            self.hidden_dim,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
        )(y)
        y = _activation(y, self.activation)
        if self.dropout_rate > 0.0:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        y = nn.Dense(
            self.output_dim,
            kernel_init=RESIDUAL_BRANCH_INIT,
            bias_init=ZERO_INIT,
        )(y)
        return residual + y


class DeepResidualMLP(ProjectorBase):
    """More expressive MLP projector for harder weak-to-strong alignment."""

    output_dim: int
    hidden_dim: int = 384
    depth: int = 3
    activation: str = "silu"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        if self.depth < 2 or self.depth > 3:
            raise ValueError(f"DeepResidualMLP depth must be in [2, 3], got {self.depth}")

        y = x
        for _ in range(self.depth):
            y = _ResidualMLPBlock(
                output_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
            )(y, train=train)
        return y


class _ConvNeXtTokenBlock(nn.Module):
    output_dim: int
    hidden_dim: int
    kernel_size: int
    activation: str
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = _project_residual(x, self.output_dim)
        y = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=True)(x)
        y = nn.Dense(
            self.hidden_dim,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
        )(y)
        y = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.kernel_size,),
            padding="SAME",
            feature_group_count=self.hidden_dim,
            kernel_init=XAVIER_UNIFORM,
            bias_init=ZERO_INIT,
        )(y)
        y = _activation(y, self.activation)
        if self.dropout_rate > 0.0:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        y = nn.Dense(
            self.output_dim,
            kernel_init=RESIDUAL_BRANCH_INIT,
            bias_init=ZERO_INIT,
        )(y)
        return residual + y


class MiniConvNeXtProjector(ProjectorBase):
    """Token-mixing projector for [B, T, C] features; 2D inputs use MLP fallback."""

    output_dim: int
    hidden_dim: int = 384
    depth: int = 1
    kernel_size: int = 3
    activation: str = "silu"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        if x.ndim == 2:
            return ResidualBottleneckMLP(
                output_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                depth=4,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
            )(x, train=train)
        if x.ndim != 3:
            raise ValueError(f"MiniConvNeXtProjector expects rank 2 or 3 input, got rank {x.ndim}")
        if self.depth < 1 or self.depth > 3:
            raise ValueError(f"MiniConvNeXtProjector depth must be in [1, 3], got {self.depth}")

        y = x
        for _ in range(self.depth):
            y = _ConvNeXtTokenBlock(
                output_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                kernel_size=self.kernel_size,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
            )(y, train=train)
        return y


def _project_residual(x: jnp.ndarray, output_dim: int) -> jnp.ndarray:
    if x.shape[-1] == output_dim:
        return x
    return nn.Dense(
        output_dim,
        kernel_init=XAVIER_UNIFORM,
        bias_init=ZERO_INIT,
        name="residual_proj",
    )(x)


def build_projector(
    kind: str = "residual_mlp",
    input_dim: int = 384,
    output_dim: int | None = None,
    hidden_dim: int = 384,
    depth: int | None = None,
    activation: str = "silu",
    dropout_rate: float = 0.0,
    kernel_size: int = 3,
) -> ProjectorBase:
    """Build a LayerSync projector module.

    Args:
        kind: One of "residual_mlp", "deep_mlp", or "convnext".
        input_dim: Used as the default output dimension.
        output_dim: Target embedding size. Defaults to input_dim.
        hidden_dim: Internal bottleneck width.
        depth: Per-kind depth. Defaults: 4, 3, and 1 respectively.
    """
    output_dim = input_dim if output_dim is None else output_dim
    kind = kind.lower()

    if kind == "residual_mlp":
        return ResidualBottleneckMLP(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=4 if depth is None else depth,
            activation=activation,
            dropout_rate=dropout_rate,
        )
    if kind == "deep_mlp":
        return DeepResidualMLP(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=3 if depth is None else depth,
            activation=activation,
            dropout_rate=dropout_rate,
        )
    if kind == "convnext":
        return MiniConvNeXtProjector(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=1 if depth is None else depth,
            kernel_size=kernel_size,
            activation=activation,
            dropout_rate=dropout_rate,
        )
    raise ValueError(
        f"Unsupported projector kind '{kind}'. "
        "Expected one of: residual_mlp, deep_mlp, convnext."
    )
