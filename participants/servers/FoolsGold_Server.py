"""
FoolsGold defense (Fung, Yoon & Beschastnikh, RAID 2020).

Reference weighting with the paper's weighted-sum aggregation
(`DistributedML/FoolsGold/ML/code/model_aggregator.py::foolsgold`).
Paper §6 (Algorithm and Indicative Features).

Per round, on the *participating* clients only:

  1. Track each client's update history when enabled.
  2. For deep models, retain output-layer parameters as indicative features.
     The paper does not use non-output-layer magnitudes to identify deep-model
     features; the original softmax-model experiment uses its entire output.
  3. Pairwise cosine similarity over filtered histories, zero diagonal:
         cs_{ij} = cos(H_i^Φ, H_j^Φ),       cs_{ii} = 0
  4. Pardoning (paper §6.2):
         v_i = max_j cs_{ij} + ε
         if v_i < v_j:   cs_{ij} ← cs_{ij} · v_i / v_j
     This rescales rows whose own max similarity is dominated by a peer's;
     prevents honest clients from being penalised because a sybil happens
     to be similar to them.
  5. Aggregation weight:
         α_i = clip(1 − max_j cs_{ij}, 0, 1)
  6. Rescale so the most-trusted client has weight 1:
         α_i ← α_i / max_k α_k
         α_i = 0.99 where α_i == 1            # avoid logit(1) → +∞
  7. Logit-confidence (paper Algorithm 1):
         α_i ← κ · (log((α_i / (1 − α_i)) + ε) + 0.5)
         clip to [0, 1]; NaN → 0, infinity → 1

The aggregation step then forms Δ = η_server Σ_i α_i Δ_i.  The authors'
reference corresponds to η_server = 1; a smaller common server step can be
used when local training produces larger updates than in their prototype.
"""

import copy
import logging
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from participants.servers.No_defense_Server import No_defense_Server
from participants.servers.defense_utils import (
    zero_like_accumulator,
)
from utils.utils import model_dist_norm_var, update_weight_accumulator

logger = logging.getLogger("logger")

# Matches `epsilon = 1e-5` in the authors' reference (model_aggregator.py).
_EPSILON = 1e-5


class FoolsGold_Server(No_defense_Server):
    def __init__(self, params, dataloader):
        # A stale submission YAML can silently select the paper's unit server
        # step while this experiment requires a ten-client FedAvg-scale step.
        # Make the experimental choices explicit before any local training.
        required = (
            "foolsgold_server_lr",
            "foolsgold_use_history",
            "foolsgold_indicative_layers",
        )
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(
                "FoolsGold YAML is missing explicit settings "
                f"{missing}; verify that the updated YAML was uploaded "
                f"(loaded path: {params.get('params', '<unknown>')})."
            )
        if type(params["foolsgold_use_history"]) is not bool:
            raise ValueError("foolsgold_use_history must be a YAML boolean")
        super().__init__(params, dataloader)
        # client_id -> 1-D tensor of summed updates restricted to indicative
        # features (only). Restricting *before* accumulation keeps memory low
        # and matches the reference impl (`summed_deltas` is full-dim there,
        # but the cosine call uses np.take(sd, sig_features_idx)).
        self.history_sum = dict()
        self._last_foolsgold_weights = None
        self.foolsgold_kappa = float(params.get("foolsgold_kappa", 1.0))
        if not math.isfinite(self.foolsgold_kappa) or self.foolsgold_kappa <= 0:
            raise ValueError("foolsgold_kappa must be finite and positive")
        self.foolsgold_server_lr = float(params["foolsgold_server_lr"])
        if not math.isfinite(self.foolsgold_server_lr) or self.foolsgold_server_lr <= 0:
            raise ValueError("foolsgold_server_lr must be finite and positive")
        self.foolsgold_use_history = params["foolsgold_use_history"]
        self.parameter_keys = [name for name, _ in self.global_model.named_parameters()]
        clip_norm = params.get("foolsgold_max_update_norm")
        self.foolsgold_max_update_norm = None if clip_norm is None else float(clip_norm)
        if self.foolsgold_max_update_norm is not None and (
            not math.isfinite(self.foolsgold_max_update_norm)
            or self.foolsgold_max_update_norm <= 0
        ):
            raise ValueError("foolsgold_max_update_norm must be finite and positive")

        # Deep-model similarity uses the output layer's indicative features.
        # "all" remains available as an explicit full-model ablation.
        configured = params["foolsgold_indicative_layers"]
        if configured == "auto":
            self.indicative_keys: Optional[List[str]] = (
                self._infer_output_layer_keys(self.global_model)
            )
            if self.indicative_keys is None:
                raise ValueError(
                    "FoolsGold could not find an nn.Linear output layer; "
                    "set foolsgold_indicative_layers explicitly"
                )
        elif configured in (None, False, "none", "all"):
            self.indicative_keys = None  # full update
        else:
            if not isinstance(configured, (list, tuple)) or not configured or not all(
                isinstance(key, str) and key for key in configured
            ):
                raise ValueError(
                    "foolsgold_indicative_layers must be 'auto', 'all', "
                    "or a nonempty list of parameter names"
                )
            self.indicative_keys = list(configured)
        logger.info(
            "[FoolsGold] effective config: source=%s server_lr=%s "
            "use_history=%s indicative_layers=%s max_update_norm=%s",
            params.get("params", "<unknown>"), self.foolsgold_server_lr,
            self.foolsgold_use_history, self.indicative_keys,
            self.foolsgold_max_update_norm,
        )

    # ──────────────────────────────────────────────────────────────
    # Indicative-feature selection
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _infer_output_layer_keys(model: nn.Module) -> Optional[List[str]]:
        """Return state_dict keys belonging to the model's last nn.Linear."""
        last_name = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                last_name = name
        if last_name is None:
            return None
        prefix = last_name + "." if last_name else ""
        keys = [k for k in model.state_dict().keys()
                if k == last_name or k.startswith(prefix)]
        if not last_name:
            keys = [key for key in ('weight', 'bias') if key in model.state_dict()]
        return keys or None

    def _flatten_indicative(self, update_dict) -> torch.Tensor:
        """Flatten only the indicative-feature entries of an update dict."""
        if not self.indicative_keys:
            keys = self.parameter_keys
        else:
            keys = self.indicative_keys
        chunks = []
        for k in keys:
            t = update_dict.get(k)
            if t is None:
                raise ValueError(f'FoolsGold indicative parameter is missing: {k}')
            if t.dtype in (torch.int64, torch.int32):
                continue
            chunks.append(t.detach().reshape(-1).float().cpu())
        if not chunks:
            raise ValueError('FoolsGold has no floating indicative features')
        return torch.cat(chunks)

    def _clip_client_update(self, update_dict) -> None:
        """Match the authors' optional unit-norm cap before history and weighting."""
        if self.foolsgold_max_update_norm is None:
            return
        squared_norm = sum(
            update_dict[name].detach().float().square().sum()
            for name in self.parameter_keys
        )
        norm = float(torch.sqrt(squared_norm).item())
        if not math.isfinite(norm):
            raise ValueError("FoolsGold received a nonfinite client update")
        if norm > self.foolsgold_max_update_norm:
            scale = self.foolsgold_max_update_norm / norm
            for name, delta in update_dict.items():
                if torch.is_floating_point(delta):
                    update_dict[name] = delta * scale

    # ──────────────────────────────────────────────────────────────
    # Core: FoolsGold weights (verbatim reference algorithm)
    # ──────────────────────────────────────────────────────────────
    def _foolsgold_weights(self, client_ids: List[int]) -> np.ndarray:
        n = len(client_ids)
        if n == 0:
            return np.zeros(0)
        if n == 1:
            # Only one participant; trust it (paper assumes ≥2 to compare).
            return np.ones(1)

        # Stack indicative-feature histories: shape (n, D_Φ).
        H = torch.stack(
            [self.history_sum[cid] for cid in client_ids], dim=0,
        ).numpy().astype(np.float64)
        if not np.isfinite(H).all():
            raise ValueError('FoolsGold history contains nonfinite updates')

        # Normalise rows for cosine similarity (matches sklearn behaviour).
        # We use sklearn here to stay 1:1 with the reference impl.
        from sklearn.metrics.pairwise import cosine_similarity
        cs = cosine_similarity(H) - np.eye(n)

        # ── Pardoning (paper §6.2) ─────────────────────────────────
        # Snapshot row-maxes BEFORE any mutation, with an additive epsilon
        # exactly as in the reference (`maxcs = np.max(cs, axis=1) + eps`).
        maxcs = cs.max(axis=1) + _EPSILON
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i, j] = cs[i, j] * maxcs[i] / maxcs[j]

        # ── Aggregation weight α_i = 1 − max_j cs_{ij}, clipped ────
        wv = 1.0 - cs.max(axis=1)
        wv[wv > 1.0] = 1.0
        wv[wv < 0.0] = 0.0

        # ── Rescale so the most-trusted client gets weight 1 ───────
        wv_max = float(wv.max())
        if wv_max > 0:
            wv = wv / wv_max
        # Sidestep logit(1) → +∞ as the reference does.
        wv[wv == 1.0] = 0.99

        # ── Logit confidence (paper Algorithm 1) ────────────────────
        with np.errstate(divide="ignore", invalid="ignore"):
            wv = self.foolsgold_kappa * (
                np.log((wv / (1.0 - wv)) + _EPSILON) + 0.5
            )

        # Clip: +inf or >1 → 1 (most-trusted), negatives → 0 (sybil), NaN → 0.
        wv[np.isinf(wv)] = 1.0
        wv[wv > 1.0] = 1.0
        wv[wv < 0.0] = 0.0
        wv = np.nan_to_num(wv, nan=0.0)
        return wv

    # ──────────────────────────────────────────────────────────────
    # Round driver
    # ──────────────────────────────────────────────────────────────
    def broadcast_upload(self, iteration, benign_client, malicious_client, **kwargs):
        logger.info(f"[FoolsGold] Training on global iteration {iteration}")
        selected_clients_list, malicious_clients_list = self.select_clients(iteration)

        global_model_copy = self.create_global_model_copy()
        global_model_snapshot = copy.deepcopy(self.global_model)

        per_client_updates = []
        update_norm_list = []

        for client_id in tqdm(selected_clients_list):
            client = malicious_client if client_id in malicious_clients_list else benign_client
            client_train_data = self.train_dataloader[client_id]

            local_model = copy.deepcopy(self.global_model)
            for _, param in local_model.named_parameters():
                param.requires_grad = True
            local_model.train()
            updated_model = client.local_train(
                iteration, local_model, client_train_data, client_id,
                test_loader=self.test_dataloader,
            )

            update_norm = model_dist_norm_var(updated_model, global_model_copy)
            update_norm_list.append(round(update_norm.item(), 6))

            tmp_acc = zero_like_accumulator(self.global_model.state_dict())
            _, single_wa = update_weight_accumulator(updated_model, global_model_snapshot, tmp_acc)
            self._clip_client_update(single_wa)
            per_client_updates.append(single_wa)
            del local_model

        # ── Update the selected client vectors; history is optional.
        for cid, upd in zip(selected_clients_list, per_client_updates):
            flat = self._flatten_indicative(upd)
            if not self.foolsgold_use_history or cid not in self.history_sum:
                self.history_sum[cid] = flat
            else:
                self.history_sum[cid] = self.history_sum[cid] + flat

        weights = self._foolsgold_weights(selected_clients_list)
        self._last_foolsgold_weights = [float(weight) for weight in weights]

        for ind, client_id in enumerate(selected_clients_list):
            tag = " (mal)" if client_id in malicious_clients_list else ""
            logger.info(
                f"[FoolsGold] Client {client_id}{tag} norm={update_norm_list[ind]} "
                f"weight={weights[ind]:.4f}"
            )

        # ── Aggregate η_server Σ_i α_i Δ_i ────────────────────────────────
        # `aggregation()` divides the accumulator by sum(aggregated_model_id).
        # Multiply by n_kept here so that division cancels, while preserving
        # the binary client IDs used by the shared diagnostics.
        weight_accumulator = self.create_weight_accumulator()
        kept = [i for i in range(len(selected_clients_list)) if weights[i] > 0]
        n_kept = len(kept)
        if n_kept == 0:
            logger.warning("[FoolsGold] all weights are zero; skipping update this round")
            return weight_accumulator, per_client_updates, [0] * len(selected_clients_list)

        weight_sum = float(weights[kept].sum())
        parameter_keys = set(self.parameter_keys)
        for i in kept:
            weight = float(weights[i])
            for name, delta in per_client_updates[i].items():
                accumulator = weight_accumulator.get(name)
                if accumulator is None or not torch.is_floating_point(accumulator):
                    continue
                # The server step changes trainable parameters only. BatchNorm
                # buffers still use a convex weighted mean of client states.
                if name in parameter_keys:
                    state_scale = n_kept * weight * self.foolsgold_server_lr
                else:
                    state_scale = n_kept * weight / weight_sum
                accumulator.add_(delta.to(accumulator) * state_scale)

        aggregated_model_id = [1 if weights[i] > 0 else 0
                               for i in range(len(selected_clients_list))]

        return weight_accumulator, per_client_updates, aggregated_model_id

    def record_detection_metrics(self, iteration, aggregation_weights):
        # Detection uses the FoolsGold weights; aggregation keeps every positive weight.
        if self._last_foolsgold_weights is None:
            raise RuntimeError("FoolsGold weights are unavailable for detection metrics")
        super().record_detection_metrics(
            iteration, self._last_foolsgold_weights, threshold=0.5
        )
