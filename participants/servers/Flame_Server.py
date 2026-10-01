import copy
import logging
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from participants.servers.BasicServer import BasicServer
from utils.utils import model_dist_norm_var

logger = logging.getLogger("logger")


class Flame_Server(BasicServer):
    """FLAME defense server.

    Implements the main FLAME pipeline from Nguyen et al.:
    cosine-distance clustering, median-norm clipping, and optional adaptive
    Gaussian noise over the accepted updates.
    """

    def __init__(self, params, dataloader):
        super(Flame_Server, self).__init__(params, dataloader)
        self.flame_eps = float(self.params.get("flame_eps", 1e-12))
        self.flame_noise_sigma = float(self.params.get("flame_noise_sigma", 0.001))
        if not math.isfinite(self.flame_noise_sigma) or self.flame_noise_sigma < 0:
            raise ValueError('flame_noise_sigma must be finite and nonnegative')
        self.flame_cluster_method = str(self.params.get("flame_cluster_method", "hdbscan")).lower()
        self.flame_min_cluster_size = self.params.get("flame_min_cluster_size", "paper")
        self.flame_allow_single_cluster = bool(self.params.get("flame_allow_single_cluster", True))
        self.param_keys = [name for name, _ in self.global_model.named_parameters()]
        self.bn_stat_keys = [
            name for name, data in self.global_model.state_dict().items()
            if torch.is_floating_point(data) and self._is_bn_running_stat(name)
        ]

    def broadcast_upload(self, iteration, benign_client, malicious_client, **kwargs):
        logger.info(f"Training on global iteration {iteration} ")

        selected_clients_list, malicious_clients_list = self.select_clients(iteration)
        global_model_copy = self.create_global_model_copy()
        global_model = copy.deepcopy(self.global_model)

        weight_accumulator_by_client = []
        local_state_by_client = []
        update_norm_list = []

        for client_id in tqdm(selected_clients_list):
            client = malicious_client if client_id in malicious_clients_list else benign_client
            client_train_data = self.train_dataloader[client_id]
            local_model = copy.deepcopy(self.global_model)

            for _, params in local_model.named_parameters():
                params.requires_grad = True

            local_model.train()
            updated_model = client.local_train(
                iteration,
                local_model,
                client_train_data,
                client_id,
                test_loader=self.test_dataloader,
            )
            update_norm = model_dist_norm_var(updated_model, global_model_copy)
            update_norm_list.append(round(update_norm.item(), 6))
            local_state_by_client.append(self._model_state(updated_model))
            weight_accumulator_by_client.append(self._model_delta(updated_model, global_model))
            del local_model

        for client_ind, client_id in enumerate(selected_clients_list):
            logger.info(f"Client {client_id} update norm: {update_norm_list[client_ind]}")

        accepted_mask, labels, cluster_label, cosine_dist = self._select_cluster(local_state_by_client)
        clipped_updates, clip_norm, scales = self._clip_updates(weight_accumulator_by_client, accepted_mask)
        weight_accumulator = self._sum_updates(clipped_updates, accepted_mask)
        self._add_bn_stat_updates(weight_accumulator, weight_accumulator_by_client, accepted_mask)
        accepted_count = int(sum(accepted_mask))
        if accepted_count > 0 and self.flame_noise_sigma > 0:
            self._add_adaptive_noise(weight_accumulator, clip_norm, accepted_count)

        aggregated_model_id = [1 if keep else 0 for keep in accepted_mask]
        logger.info(
            "FLAME clustering: method=%s, labels=%s, selected_cluster=%s, accepted=%s",
            self.flame_cluster_method,
            labels,
            cluster_label,
            aggregated_model_id,
        )
        logger.info(
            "FLAME clipping: clip_norm=%.6f, noise_sigma=%.6f, scales=%s",
            float(clip_norm),
            self.flame_noise_sigma,
            [round(float(scale), 6) for scale in scales],
        )
        if cosine_dist is not None:
            logger.info(
                "FLAME cosine distance summary: min=%.6f, median=%.6f, max=%.6f",
                float(cosine_dist.min().item()),
                float(cosine_dist.median().item()),
                float(cosine_dist.max().item()),
            )

        return weight_accumulator, weight_accumulator_by_client, aggregated_model_id

    def _model_delta(self, model, global_model):
        delta = {}
        global_state = global_model.state_dict()
        for name, data in model.state_dict().items():
            delta[name] = (data.detach() - global_state[name].detach()).clone()
        return delta

    def _model_state(self, model):
        return {
            name: data.detach().cpu().clone()
            for name, data in model.state_dict().items()
        }

    def _is_bn_running_stat(self, name):
        return "running_mean" in name or "running_var" in name

    def _flatten_tensors(self, tensor_dict, keys):
        chunks = []
        for name in keys:
            tensor = tensor_dict.get(name)
            if tensor is not None and torch.is_floating_point(tensor):
                chunks.append(tensor.detach().float().reshape(-1).cpu())
        if not chunks:
            return torch.zeros(1)
        return torch.cat(chunks)

    def _flatten_model_state(self, state):
        return self._flatten_tensors(state, self.param_keys)

    def _flatten_update(self, update):
        return self._flatten_tensors(update, self.param_keys)

    def _select_cluster(self, local_states):
        n_updates = len(local_states)
        flat_models = torch.stack([self._flatten_model_state(state) for state in local_states])
        finite_mask = torch.isfinite(flat_models).all(dim=1)
        finite_indices = torch.nonzero(finite_mask, as_tuple=False).flatten().tolist()
        rejected_indices = [idx for idx in range(n_updates) if idx not in finite_indices]
        if rejected_indices:
            logger.info(f"FLAME rejected non-finite local models before clustering: {rejected_indices}")

        if n_updates <= 1:
            return [bool(finite_mask[0].item())] * n_updates, [0 if finite_indices else -2] * n_updates, 0, None

        min_cluster_size = self._hdbscan_min_cluster_size(n_updates)
        if len(finite_indices) < min_cluster_size:
            labels = [-2 if idx in rejected_indices else -3 for idx in range(n_updates)]
            logger.info(
                "FLAME rejected round: only %d/%d finite updates, below min_cluster_size=%d",
                len(finite_indices),
                n_updates,
                min_cluster_size,
            )
            return [False] * n_updates, labels, "not_enough_finite_updates", None

        finite_flat_models = flat_models[finite_indices]
        cosine_dist = self._cosine_distance(finite_flat_models)
        finite_labels = self._cluster_updates(cosine_dist, total_updates=n_updates)
        finite_accepted_mask, cluster_label = self._largest_cluster_mask(finite_labels)

        labels = [-2] * n_updates
        accepted_mask = [False] * n_updates
        for local_idx, original_idx in enumerate(finite_indices):
            labels[original_idx] = finite_labels[local_idx]
            accepted_mask[original_idx] = finite_accepted_mask[local_idx]


        return accepted_mask, labels, cluster_label, cosine_dist

    def _cosine_distance(self, flat_updates):
        normalized = F.normalize(flat_updates, p=2, dim=1, eps=self.flame_eps)
        cosine_sim = torch.mm(normalized, normalized.t()).clamp(-1.0, 1.0)
        cosine_dist = (1.0 - cosine_sim).clamp(min=0.0)
        cosine_dist.fill_diagonal_(0.0)
        return cosine_dist

    def _cluster_updates(self, cosine_dist, total_updates=None):
        method = self.flame_cluster_method
        if method != "hdbscan":
            raise ValueError("FLAME paper baseline uses HDBSCAN; set flame_cluster_method: hdbscan")
        return self._cluster_hdbscan(cosine_dist, total_updates=total_updates)

    def _cluster_hdbscan(self, cosine_dist, total_updates=None):
        try:
            import hdbscan
        except ImportError as exc:
            raise ImportError("FLAME requires the hdbscan package for paper-faithful clustering") from exc

        n_samples = int(cosine_dist.shape[0])
        min_cluster_size = self._hdbscan_min_cluster_size(total_updates or n_samples)
        logger.info(f"FLAME HDBSCAN min_cluster_size={min_cluster_size}")
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=min_cluster_size,
            min_samples=int(self.params.get("flame_min_samples", 1)),
            allow_single_cluster=self.flame_allow_single_cluster,
        )
        distance_np = cosine_dist.detach().cpu().double().contiguous().numpy()
        return clusterer.fit_predict(distance_np).tolist()

    def _hdbscan_min_cluster_size(self, total_updates):
        configured = self.flame_min_cluster_size
        if configured is None or str(configured).lower() in ("paper", "majority", "auto"):
            min_cluster_size = (int(total_updates) // 2) + 1
        else:
            min_cluster_size = max(2, int(configured))
        return min_cluster_size

    def _largest_cluster_mask(self, labels):
        counts = {}
        for label in labels:
            if label == -1:
                continue
            counts[label] = counts.get(label, 0) + 1

        if not counts:
            return [False] * len(labels), "all_noise_reject_all"

        largest_label = max(counts, key=counts.get)
        return [label == largest_label for label in labels], largest_label

    def _clip_updates(self, updates, accepted_mask):
        update_norms = [
            torch.norm(self._flatten_update(update), p=2)
            for update in updates
        ]
        finite_norms = [norm for norm in update_norms if torch.isfinite(norm)]

        if finite_norms:
            clip_norm = torch.quantile(torch.stack(finite_norms), 0.5).item()
        else:
            clip_norm = 0.0

        if "flame_clip_norm" in self.params:
            clip_norm = float(self.params["flame_clip_norm"])

        clipped_updates = []
        scales = []
        for update, norm_tensor in zip(updates, update_norms):
            norm = norm_tensor.item()
            scale = min(1.0, clip_norm / (norm + self.flame_eps)) if clip_norm > 0 and math.isfinite(norm) else 0.0
            scales.append(scale)
            clipped_updates.append(self._scale_update(update, scale))

        return clipped_updates, clip_norm, scales

    def _scale_update(self, update, scale):
        scaled_update = {}
        for name, delta in update.items():
            if name in self.bn_stat_keys:
                scaled_update[name] = torch.zeros_like(delta)
            elif torch.is_floating_point(delta):
                scaled_update[name] = torch.zeros_like(delta) if scale == 0.0 else delta * scale
            else:
                scaled_update[name] = torch.zeros_like(delta)
        return scaled_update

    def _sum_updates(self, updates, accepted_mask):
        weight_accumulator = self.create_weight_accumulator()
        for keep, update in zip(accepted_mask, updates):
            if not keep:
                continue
            for name in weight_accumulator.keys():
                if name in update and torch.is_floating_point(weight_accumulator[name]):
                    weight_accumulator[name].add_(update[name].to(weight_accumulator[name].device))
        return weight_accumulator

    def _add_bn_stat_updates(self, weight_accumulator, updates, accepted_mask):
        accepted_count = int(sum(accepted_mask))
        if accepted_count == 0:
            return
        for update, keep in zip(updates, accepted_mask):
            if not keep:
                continue
            for name in self.bn_stat_keys:
                if name in weight_accumulator and name in update:
                    weight_accumulator[name].add_(update[name].to(weight_accumulator[name].device))

    def _add_adaptive_noise(self, weight_accumulator, clip_norm, accepted_count):
        noise_std = self.flame_noise_sigma * float(clip_norm) * max(1, accepted_count)
        for name, data in weight_accumulator.items():
            if torch.is_floating_point(data) and name not in self.bn_stat_keys:
                data.add_(torch.normal(mean=0.0, std=noise_std, size=data.shape, device=data.device))
