"""Krum and Multi-Krum (Blanchard et al., NeurIPS 2017).

Score for client i is the sum of squared distances to its (n - f - 2)
nearest neighbours, where f is the assumed number of Byzantine workers.
Krum picks the single client with the smallest score; Multi-Krum picks
the m smallest scores and averages them.
"""
import copy
import logging

import numpy as np
import torch
from tqdm import tqdm

from participants.servers.No_defense_Server import No_defense_Server
from participants.servers.defense_utils import (
    add_scaled_update,
    pairwise_sq_distances,
    stack_updates,
    zero_like_accumulator,
)
from utils.utils import model_dist_norm_var, update_weight_accumulator

logger = logging.getLogger("logger")


def _krum_scores(flat_stack: torch.Tensor, f: int) -> np.ndarray:
    if flat_stack.ndim != 2 or not torch.isfinite(flat_stack).all():
        raise ValueError('Krum requires a finite two-dimensional update matrix')
    n = flat_stack.shape[0]
    if isinstance(f, bool) or int(f) != f or f < 0 or n <= 2 * f + 2:
        raise ValueError('Krum requires integer f >= 0 and n > 2*f + 2')
    dists = pairwise_sq_distances(flat_stack)
    # Mask out self-distance.
    dists.fill_diagonal_(float("inf"))
    k = n - int(f) - 2
    sorted_d, _ = torch.sort(dists, dim=1)
    return sorted_d[:, :k].sum(dim=1).cpu().numpy()


class _KrumLikeServer(No_defense_Server):
    """Common per-round runner that defers selection to a subclass hook."""

    def _select_indices(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _label(self) -> str:
        return "Krum-like"

    def _byzantine_count(self, n_selected: int) -> int:
        budget = self.params.get('krum_byzantine', self.params.get('no_of_adversaries', 1))
        if isinstance(budget, bool) or int(budget) != budget or budget < 0:
            raise ValueError('krum_byzantine must be a nonnegative integer')
        if n_selected <= 2 * budget + 2:
            raise ValueError(f'Krum requires n > 2*f + 2; got n={n_selected}, f={budget}')
        return int(budget)

    def broadcast_upload(self, iteration, benign_client, malicious_client, **kwargs):
        label = self._label()
        logger.info(f"[{label}] Training on global iteration {iteration}")
        selected_clients_list, malicious_clients_list = self.select_clients(iteration)

        global_model_copy = self.create_global_model_copy()
        global_model_snapshot = copy.deepcopy(self.global_model)

        per_client_updates = []
        update_norm_list = []

        for client_id in tqdm(selected_clients_list):
            client = malicious_client if client_id in malicious_clients_list else benign_client
            client_train_data = self.train_dataloader[client_id]

            local_model = copy.deepcopy(self.global_model)
            for _, params in local_model.named_parameters():
                params.requires_grad = True
            local_model.train()
            updated_model = client.local_train(
                iteration, local_model, client_train_data, client_id,
                test_loader=self.test_dataloader,
            )

            update_norm = model_dist_norm_var(updated_model, global_model_copy)
            update_norm_list.append(round(update_norm.item(), 6))

            tmp_acc = zero_like_accumulator(self.global_model.state_dict())
            _, single_wa = update_weight_accumulator(updated_model, global_model_snapshot, tmp_acc)
            per_client_updates.append(single_wa)
            del local_model

        parameter_keys = [name for name, _ in self.global_model.named_parameters()]
        flat_stack = stack_updates([
            {name: update[name] for name in parameter_keys} for update in per_client_updates
        ])
        f = self._byzantine_count(len(selected_clients_list))
        scores = _krum_scores(flat_stack, f)
        chosen = self._select_indices(scores)

        weight_accumulator = self.create_weight_accumulator()
        if len(chosen) == 0:
            logger.warning(f"[{label}] no clients selected; skipping update")
            return weight_accumulator, per_client_updates, [0] * len(selected_clients_list)

        for idx in chosen:
            add_scaled_update(weight_accumulator, per_client_updates[idx], 1.0)

        aggregated_model_id = [0] * len(selected_clients_list)
        for idx in chosen:
            aggregated_model_id[idx] = 1

        for ind, client_id in enumerate(selected_clients_list):
            tag = " (mal)" if client_id in malicious_clients_list else ""
            picked = "*" if ind in chosen else " "
            logger.info(
                f"[{label}] {picked} Client {client_id}{tag} norm={update_norm_list[ind]} score={scores[ind]:.4f}"
            )
        logger.info(f"[{label}] f={f} kept={len(chosen)}/{len(selected_clients_list)}")
        return weight_accumulator, per_client_updates, aggregated_model_id


class Krum_Server(_KrumLikeServer):
    def _label(self) -> str:
        return "Krum"

    def _select_indices(self, scores: np.ndarray) -> np.ndarray:
        return np.array([int(np.argmin(scores))])


class MultiKrum_Server(_KrumLikeServer):
    def _label(self) -> str:
        return "MultiKrum"

    def _select_indices(self, scores: np.ndarray) -> np.ndarray:
        n = len(scores)
        f = self._byzantine_count(n)
        # Default to keeping (n - f) safest clients.
        configured = self.params.get('multikrum_m', n - f)
        if isinstance(configured, bool) or int(configured) != configured or not 1 <= configured <= n:
            raise ValueError('multikrum_m must be an integer in [1,n]')
        m = int(configured)
        return np.argsort(scores)[:m]
