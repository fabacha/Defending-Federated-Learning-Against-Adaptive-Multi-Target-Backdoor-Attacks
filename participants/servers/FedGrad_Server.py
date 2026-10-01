"""FedGrad adapted from the authors' released implementation to SATML's server API.

Reference: https://github.com/judydnguyen/FedGrad_Backdoor_Attack/blob/main/defense.py
The hard-filter switch is counted in observed defense rounds so a run resumed at
iteration 2000 first gathers the pair history needed by the hard filter.
"""

import copy
import logging
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from participants.servers.BasicServer import BasicServer

logger = logging.getLogger("logger")


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def _minmax(values):
    values = np.asarray(values, dtype=np.float64)
    span = float(values.max() - values.min())
    return np.zeros_like(values) if span <= 1e-12 else (values - values.min()) / span


def _column_minmax(matrix):
    """Match MinMaxScaler.fit_transform on each round's similarity matrix."""
    low = matrix.min(axis=0)
    span = matrix.max(axis=0) - low
    return np.divide(matrix - low, span, out=np.zeros_like(matrix), where=span > 1e-12)


class FedGrad_Server(BasicServer):
    def __init__(self, params, dataloader):
        super().__init__(params, dataloader)
        self.zeta = float(params.get("fedgrad_zeta", 0.5))
        self.gamma = float(params.get("fedgrad_trust_threshold", 0.75))
        self.lambda_bad = float(params.get("fedgrad_trust_bad", 0.25))
        self.lambda_good = float(params.get("fedgrad_trust_good", 1.0))
        self.hard_warmup = int(params.get("fedgrad_hard_warmup_rounds", 10))
        self.use_trustworthy = bool(params.get("fedgrad_use_trustworthy", True))
        total = int(params["no_of_total_participants"])
        per_round = int(params["no_of_participants_per_iteration"])
        adversaries = int(params.get("no_of_adversaries", 0))
        # The authors use int(attacker_percent * part_nets_per_round).
        inferred_attackers = int(adversaries / total * per_round)
        self.estimated_attackers = int(params.get(
            "fedgrad_estimated_attackers_per_round", inferred_attackers))
        if not (0 <= self.zeta <= 1 and self.lambda_bad <= self.gamma <= self.lambda_good):
            raise ValueError("Invalid FedGrad thresholds or trust scores")
        if self.hard_warmup < 0 or self.estimated_attackers < 0:
            raise ValueError("FedGrad warmup and attacker estimate must be nonnegative")
        heads = [(name, layer) for name, layer in self.global_model.named_modules()
                 if isinstance(layer, torch.nn.Linear)]
        if not heads or heads[-1][1].bias is None:
            raise ValueError("FedGrad requires a final Linear classifier with bias")
        prefix = heads[-1][0] + "." if heads[-1][0] else ""
        self.weight_key, self.bias_key = prefix + "weight", prefix + "bias"
        self.param_keys = [name for name, _ in self.global_model.named_parameters()]
        self.rounds_seen = 0
        self.compromise_mean = {}
        self.compromise_count = defaultdict(int)
        self.trust_mean = {}
        self.trust_count = defaultdict(int)
        self.pair_mean = {}
        self.pair_count = defaultdict(int)
        self.last_fedgrad = {}

    def _soft_filter(self, ids, deltas, sample_counts):
        sample_weights = np.asarray(sample_counts, dtype=np.float64)
        sample_weights /= sample_weights.sum()
        weight_updates = [d[self.weight_key].double().numpy() for d in deltas]
        mean_update = sum(float(w) * d for w, d in zip(sample_weights, weight_updates))
        # Released code: cosine(W_i - W_average, W_average - W_global).
        instant = np.array([_cosine(d - mean_update, mean_update)
                            for d in weight_updates])
        normalized = _minmax(instant)
        for client_id, value in zip(ids, normalized):
            self.compromise_count[client_id] += 1
            n = self.compromise_count[client_id]
            old = self.compromise_mean.get(client_id, 0.0)
            self.compromise_mean[client_id] = old + (float(value) - old) / n
        threshold = min(self.zeta, float(np.median([self.compromise_mean[i] for i in ids])))
        suspicious = {i for i in ids if self.compromise_mean[i] > threshold}
        return suspicious, threshold, mean_update

    def _hard_filter(self, ids, deltas, mean_weight_update, global_bias, active):
        n = len(ids)
        weight_rows = [(d[self.weight_key].double().numpy() - mean_weight_update).sum(axis=1)
                       for d in deltas]
        # The released implementation uses the local bias values, not bias deltas.
        bias_rows = [(global_bias + d[self.bias_key].double().numpy()).reshape(-1)
                     for d in deltas]
        round_weight = np.array([[_cosine(a, b) for b in weight_rows]
                                 for a in weight_rows], dtype=np.float64)
        round_bias = np.array([[_cosine(a, b) for b in bias_rows]
                               for a in bias_rows], dtype=np.float64)
        round_weight = _column_minmax(round_weight)
        round_bias = _column_minmax(round_bias)

        features = np.zeros((n, 2 * n), dtype=np.float64)
        for a in range(n):
            for b in range(n):
                # Column-wise scaling makes i->j and j->i generally different.
                key = (ids[a], ids[b])
                current = np.array([round_weight[a, b], round_bias[a, b]])
                count = self.pair_count[key] + 1
                previous = self.pair_mean.get(key, np.zeros(2))
                cumulative = previous + (current - previous) / count
                self.pair_mean[key], self.pair_count[key] = cumulative, count
                features[a, b] = cumulative[0]
                features[a, n + b] = cumulative[1]

        neighbours = n - self.estimated_attackers - 2
        if not active or n < 3 or neighbours < 1 or np.allclose(features, features[0]):
            return set()
        labels = KMeans(n_clusters=2, n_init=10).fit_predict(features)
        # The authors' closeness score uses squared distances over all model parameters.
        distances = np.zeros((n, n), dtype=np.float64)
        for a in range(n):
            for b in range(a + 1, n):
                squared = sum(float(torch.sum((deltas[a][key].double() -
                                               deltas[b][key].double()).square()))
                              for key in self.param_keys)
                distances[a, b] = distances[b, a] = squared
        closeness = [np.sort(np.delete(distances[a], a))[:neighbours].sum()
                     for a in range(n)]
        benign_cluster = int(labels[int(np.argmin(closeness))])
        return {client_id for client_id, label in zip(ids, labels)
                if label != benign_cluster}

    def _filter(self, ids, deltas, sample_counts, global_bias):
        soft, threshold, mean_update = self._soft_filter(ids, deltas, sample_counts)
        # Gather similarity history from the first defense round, as in the release.
        hard_active = self.rounds_seen + 1 >= self.hard_warmup
        hard = self._hard_filter(ids, deltas, mean_update, global_bias, hard_active)
        provisional = soft | hard
        if hard_active and self.use_trustworthy:
            # The released code consults the trust history BEFORE appending this round.
            trusted_filter = {i for i in provisional
                              if self.trust_mean.get(i, 0.5) < self.gamma}
            rejected = trusted_filter if trusted_filter else set(soft)
        else:
            rejected = set(provisional)

        for client_id in ids:
            previous = self.trust_mean.get(client_id, 0.5)
            count = self.trust_count[client_id] + 1  # Include the authors' initial 0.5.
            instant = self.lambda_bad if client_id in rejected else self.lambda_good
            self.trust_mean[client_id] = previous + (instant - previous) / (count + 1)
            self.trust_count[client_id] += 1
        self.last_fedgrad = {"soft": sorted(soft), "hard": sorted(hard),
                             "rejected": sorted(rejected), "threshold": threshold,
                             "trust": {i: self.trust_mean[i] for i in ids},
                             "hard_active": hard_active}
        self.rounds_seen += 1
        return rejected

    def broadcast_upload(self, iteration, benign_client, malicious_client, **kwargs):
        ids, malicious = self.select_clients(iteration)
        base = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
        deltas, counts = [], []
        for client_id in tqdm(ids):
            client = malicious_client if client_id in malicious else benign_client
            local = copy.deepcopy(self.global_model).train()
            local.requires_grad_(True)
            updated = client.local_train(iteration, local, self.train_dataloader[client_id],
                                         client_id, test_loader=self.test_dataloader)
            state = updated.state_dict()
            deltas.append({name: (value.detach().cpu() - base[name]) for name, value in state.items()})
            counts.append(len(self.train_dataloader[client_id].dataset))
            del local, updated
        if not all(count > 0 for count in counts):
            raise ValueError("FedGrad requires nonempty client datasets")
        rejected = self._filter(ids, deltas, counts, base[self.bias_key].double().numpy())
        accumulator = self.create_weight_accumulator()
        for client_id, delta, count in zip(ids, deltas, counts):
            if client_id not in rejected:
                for name, value in delta.items():
                    accumulator[name].add_(value.to(accumulator[name]) * count)
        weights = [count if i not in rejected else 0 for i, count in zip(ids, counts)]
        # Keep sample counts for the weighted aggregation, but expose the
        # inclusion decision separately for the shared aggregated_model log.
        self.last_aggregation_mask = [int(weight > 0) for weight in weights]
        logger.info("FedGrad round=%s %s", iteration, self.last_fedgrad)
        logger.info("FedGrad aggregation sample_weights_by_client=%s",
                    dict(zip(ids, weights)))
        return accumulator, deltas, weights
