"""Shared helpers for defense aggregations.

These helpers operate on the per-client update dicts produced by
``utils.utils.update_weight_accumulator`` (i.e. ``client_state - global_state``).
"""
import logging
from typing import Dict, List

import torch

logger = logging.getLogger("logger")


def flatten_update(update: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a state-dict-shaped update to a single 1-D float tensor.

    Skips int tensors (BN running counters etc.) so the resulting vector only
    contains the trainable / float parameters that defenses reason about.
    """
    pieces = []
    for name, tensor in update.items():
        if tensor.dtype == torch.int64 or tensor.dtype == torch.int32:
            continue
        pieces.append(tensor.detach().reshape(-1).float())
    return torch.cat(pieces)


def stack_updates(updates: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    return torch.stack([flatten_update(u) for u in updates], dim=0)


def zero_like_accumulator(reference: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = dict()
    for name, tensor in reference.items():
        out[name] = torch.zeros_like(tensor)
    return out


def add_scaled_update(accumulator: Dict[str, torch.Tensor],
                      update: Dict[str, torch.Tensor],
                      scale: float) -> None:
    """In-place ``accumulator += scale * update`` over matching keys."""
    for name, tensor in update.items():
        if name not in accumulator:
            continue
        if tensor.dtype == torch.int64 or tensor.dtype == torch.int32:
            accumulator[name].add_(tensor)
            continue
        accumulator[name].add_(tensor.float() * scale)


def update_l2_norm(update: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.linalg.norm(flatten_update(update))


def pairwise_sq_distances(matrix: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix of squared L2 distances between rows of ``matrix``."""
    sq = (matrix * matrix).sum(dim=1)
    dists = sq.unsqueeze(0) + sq.unsqueeze(1) - 2.0 * (matrix @ matrix.T)
    return dists.clamp(min=0.0)


def pairwise_cosine_similarity(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norms = matrix.norm(dim=1, keepdim=True).clamp(min=eps)
    normalized = matrix / norms
    return normalized @ normalized.T
