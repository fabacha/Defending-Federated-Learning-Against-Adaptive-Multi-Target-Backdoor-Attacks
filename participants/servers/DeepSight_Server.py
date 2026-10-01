"""DeepSight (Rieger et al., NDSS 2022) for the existing FedAvg server API.

Paper: https://www.ndss-symposium.org/wp-content/uploads/2022-156-paper.pdf
The paper does not specify the random probe distribution or HDBSCAN size
parameters; both are explicit experiment settings below.
"""

import copy
import logging
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from participants.servers.BasicServer import BasicServer


logger = logging.getLogger("logger")


def normalized_update_energy(weight_delta, bias_delta):
    """Equations 7-8: squared L1 energy per output neuron, normalized."""
    energy = weight_delta.detach().double().abs().flatten(1).sum(1)
    energy = energy + bias_delta.detach().double().abs()
    squared = energy.square()
    total = squared.sum()
    if not torch.isfinite(total):
        raise ValueError("DeepSight received nonfinite output-layer updates")
    if total == 0:
        return torch.zeros_like(energy)
    return squared / total


def threshold_exceedings(neup):
    """Equations 9-11, using a strict exceeding comparison."""
    factor = max(0.01, 1.0 / neup.numel())
    return int((neup > factor * neup.max()).sum().item())


def suspicious_labels(exceedings):
    """Algorithm 1: suspicious when TE <= half the round median."""
    boundary = float(np.median(exceedings)) / 2.0
    return [int(value <= boundary) for value in exceedings]


def cluster_distance(labels):
    """Algorithm 2's binary co-clustering distances; noise is singleton."""
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    distances = (labels[:, None] != labels[None, :]).astype(np.float64)
    noise = labels == -1
    distances[np.logical_and(noise[:, None], noise[None, :])] = 1.0
    np.fill_diagonal(distances, 0.0)
    return distances


def cluster_vote(labels, suspicious, threshold=1.0 / 3.0):
    """Keep a cluster iff its fraction of suspicious updates is < tau."""
    groups = {}
    for index, label in enumerate(labels):
        key = ("noise", index) if label == -1 else ("cluster", int(label))
        groups.setdefault(key, []).append(index)
    accepted = [False] * len(labels)
    for indices in groups.values():
        if sum(suspicious[index] for index in indices) / len(indices) < threshold:
            for index in indices:
                accepted[index] = True
    return accepted


class DeepSight_Server(BasicServer):
    def __init__(self, params, dataloader):
        super().__init__(params, dataloader)
        linear_layers = [(name, module) for name, module in self.global_model.named_modules()
                         if isinstance(module, nn.Linear)]
        if not linear_layers or linear_layers[-1][1].bias is None:
            raise ValueError("DeepSight needs an output nn.Linear layer with a bias")
        output_name = linear_layers[-1][0]
        prefix = f"{output_name}." if output_name else ""
        self.output_weight_key = f"{prefix}weight"
        self.output_bias_key = f"{prefix}bias"
        self.parameter_keys = {name for name, _ in self.global_model.named_parameters()}
        self.input_shape = self._input_shape()
        self.probe_samples = int(params.get("deepsight_probe_samples", 20000))
        self.probe_batch_size = int(params.get("deepsight_probe_batch_size", 128))
        self.seeds = tuple(params.get("deepsight_probe_seeds", [11, 23, 37]))
        self.min_cluster_size = int(params.get("deepsight_min_cluster_size", 2))
        self.min_samples = int(params.get("deepsight_min_samples", 1))
        if self.probe_samples < 1 or self.probe_batch_size < 1 or len(self.seeds) != 3:
            raise ValueError("DeepSight needs positive probe counts and exactly three seeds")
        if self.min_cluster_size < 2 or self.min_samples < 1:
            raise ValueError("DeepSight HDBSCAN sizes must be positive; min_cluster_size >= 2")
        self.final_cluster_models = {}
        self.final_client_clusters = {}

    def _input_shape(self):
        configured = self.params.get("deepsight_input_shape")
        if configured is not None:
            return tuple(int(value) for value in configured)
        convolutions = [module for module in self.global_model.modules()
                        if isinstance(module, nn.Conv2d)]
        if convolutions:
            return (convolutions[0].in_channels, 32, 32)
        raise ValueError("Set deepsight_input_shape for a model without Conv2d")

    @staticmethod
    def _delta(local_state, global_state):
        return {name: (local_state[name] - base).detach().cpu().clone()
                for name, base in global_state.items()}

    def _output_features(self, updates):
        neups = []
        exceedings = []
        bias_updates = []
        for update in updates:
            weight = update[self.output_weight_key]
            bias = update[self.output_bias_key]
            neup = normalized_update_energy(weight, bias)
            neups.append(neup.numpy())
            exceedings.append(threshold_exceedings(neup))
            bias_updates.append(bias.detach().double().flatten())
        bias_vectors = torch.stack(bias_updates)
        normalized = F.normalize(bias_vectors, p=2, dim=1)
        cosine = (1.0 - normalized @ normalized.T).clamp(0.0, 2.0).numpy()
        np.fill_diagonal(cosine, 0.0)
        return np.stack(neups), exceedings, cosine

    def _probe_batch(self, generator, count, device):
        # The paper specifies random inputs, but not their distribution.
        # Uniform [0,1] image-valued inputs are an explicit choice here.
        return torch.rand((count, *self.input_shape), generator=generator).to(device)

    @torch.inference_mode()
    def _division_differences(self, local_states, global_model):
        """Equation 6, with the same probes for every client and global model."""
        device = next(global_model.parameters()).device
        global_model.eval()
        result = []
        for seed in self.seeds:
            ddifs = []
            for local_state in local_states:
                local_model = copy.deepcopy(global_model)
                local_model.load_state_dict(local_state)
                local_model.eval()
                generator = torch.Generator(device="cpu").manual_seed(int(seed))
                ratio_sum = None
                for start in range(0, self.probe_samples, self.probe_batch_size):
                    count = min(self.probe_batch_size, self.probe_samples - start)
                    probes = self._probe_batch(generator, count, device)
                    log_global = F.log_softmax(global_model(probes).double(), dim=1)
                    log_local = F.log_softmax(local_model(probes).double(), dim=1)
                    ratios = torch.exp(log_local - log_global)
                    batch_sum = ratios.sum(dim=0)
                    ratio_sum = batch_sum if ratio_sum is None else ratio_sum + batch_sum
                ddif = (ratio_sum / self.probe_samples).cpu().numpy()
                if not np.isfinite(ddif).all():
                    raise ValueError("DeepSight DDifs became nonfinite")
                ddifs.append(ddif)
                del local_model
            result.append(np.stack(ddifs))
        return result

    def _hdbscan(self, values, precomputed=False):
        import hdbscan

        n = len(values)
        if n == 1:
            return np.array([0], dtype=int)
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed" if precomputed else "euclidean",
            min_cluster_size=min(self.min_cluster_size, n),
            min_samples=self.min_samples,
            allow_single_cluster=True,
        )
        data = np.asarray(values, dtype=np.float64)
        if precomputed:
            data = np.maximum(data, 0.0)
            np.fill_diagonal(data, 0.0)
        return clusterer.fit_predict(data)

    def _ensemble_clusters(self, neups, ddifs, cosine):
        """Algorithm 2: five first-stage clusterings and final HDBSCAN."""
        if len(neups) == 1:
            return np.array([0], dtype=int)
        cosine_dist = cluster_distance(self._hdbscan(cosine, precomputed=True))
        neup_dist = cluster_distance(self._hdbscan(neups))
        ddif_dist = np.mean([cluster_distance(self._hdbscan(values))
                             for values in ddifs], axis=0)
        merged = (cosine_dist + neup_dist + ddif_dist) / 3.0
        return self._hdbscan(merged, precomputed=True)

    def _clip(self, updates):
        """Equation 12: median norm over every submitted update."""
        norms = []
        for update in updates:
            squared = sum(update[key].double().square().sum().item()
                          for key in self.parameter_keys)
            norm = math.sqrt(squared)
            if not math.isfinite(norm):
                raise ValueError("DeepSight received a nonfinite update")
            norms.append(norm)
        median = float(np.median(norms))
        scales = [min(1.0, median / norm) if norm else 1.0 for norm in norms]
        return median, scales

    def _sum_updates(self, updates, indices, scales):
        accumulator = self.create_weight_accumulator()
        for index in indices:
            for name, destination in accumulator.items():
                if not torch.is_floating_point(destination):
                    continue
                # BN running statistics are model state, not trainable weights.
                scale = scales[index] if name in self.parameter_keys else 1.0
                destination.add_(updates[index][name].to(destination) * scale)
        return accumulator

    def _build_final_cluster_models(self, iteration, client_ids, labels, updates, scales, global_state):
        if iteration != self.params["end_iteration"] - 1:
            return
        groups = {}
        for index, label in enumerate(labels):
            key = f"noise_{index}" if label == -1 else str(int(label))
            groups.setdefault(key, []).append(index)
            self.final_client_clusters[client_ids[index]] = key
        for key, indices in groups.items():
            state = {}
            for name, base in global_state.items():
                if torch.is_floating_point(base):
                    delta = sum((updates[index][name].to(base) *
                                 (scales[index] if name in self.parameter_keys else 1.0))
                                for index in indices) / len(indices)
                    state[name] = (base + delta).detach().cpu().clone()
                else:
                    state[name] = base.detach().cpu().clone()
            self.final_cluster_models[key] = state

    def broadcast_upload(self, iteration, benign_client, malicious_client, **kwargs):
        selected, malicious = self.select_clients(iteration)
        global_state = {name: tensor.detach().cpu().clone()
                        for name, tensor in self.global_model.state_dict().items()}
        global_model = copy.deepcopy(self.global_model).eval()
        local_states = []
        updates = []
        for client_id in tqdm(selected):
            client = malicious_client if client_id in malicious else benign_client
            local_model = copy.deepcopy(self.global_model).train()
            for parameter in local_model.parameters():
                parameter.requires_grad_(True)
            trained = client.local_train(
                iteration, local_model, self.train_dataloader[client_id], client_id,
                test_loader=self.test_dataloader,
            )
            state = {name: tensor.detach().cpu().clone()
                     for name, tensor in trained.state_dict().items()}
            local_states.append(state)
            updates.append(self._delta(state, global_state))
            del local_model, trained

        neups, exceedings, cosine = self._output_features(updates)
        ddifs = self._division_differences(local_states, global_model)
        labels = self._ensemble_clusters(neups, ddifs, cosine)
        suspicious = suspicious_labels(exceedings)
        accepted = cluster_vote(labels, suspicious)
        median, scales = self._clip(updates)
        accumulator = self._sum_updates(updates, [i for i, keep in enumerate(accepted) if keep], scales)
        self._build_final_cluster_models(iteration, selected, labels, updates, scales, global_state)
        logger.info("DeepSight: TE=%s suspicious=%s clusters=%s accepted=%s median_norm=%.6g",
                    exceedings, suspicious, labels.tolist(), accepted, median)
        return accumulator, updates, [int(keep) for keep in accepted]

    def save_model(self, iteration, trigger_set, mask_set):
        super().save_model(iteration, trigger_set, mask_set)
        if iteration == self.params["end_iteration"] - 1 and self.final_cluster_models:
            destination = Path(self.params["folder_path"]) / "DeepSight_personalized_final.pt.tar"
            torch.save({"iteration": iteration,
                        "cluster_models": self.final_cluster_models,
                        "client_clusters": self.final_client_clusters}, destination)
            logger.info("DeepSight final cluster models saved to %s", destination)

    def personalized_model_for_client(self, client_id):
        """Return the paper's final-round cluster model for a participating client."""
        cluster = self.final_client_clusters[client_id]
        model = copy.deepcopy(self.global_model)
        model.load_state_dict(self.final_cluster_models[cluster])
        return model
