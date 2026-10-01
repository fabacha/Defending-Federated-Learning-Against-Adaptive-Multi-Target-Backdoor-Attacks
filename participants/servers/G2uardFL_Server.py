import collections
import copy
import logging
import math
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from participants.servers.BasicServer import BasicServer
from utils.utils import model_dist_norm_var


logger = logging.getLogger("logger")


def _vec(params):
    return np.concatenate([value.reshape(-1) for value in params])


def _cos(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)
        self.activation = activation

    def forward(self, nodes, adjacency):
        return self.activation(adjacency @ (nodes @ self.weight))


class ClusterAssignment(nn.Module):
    """Official GuardFL Student-t clustering assignment."""

    def __init__(self, n_clusters, embedding_dim, alpha, centers=None):
        super().__init__()
        initial = torch.empty(n_clusters, embedding_dim) if centers is None else centers
        if centers is None:
            nn.init.xavier_normal_(initial)
        self.cluster_centers = nn.Parameter(initial)
        self.alpha = alpha

    def forward(self, embeddings):
        squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, dim=2)
        numerator = (1.0 / (1.0 + squared / self.alpha)) ** ((self.alpha + 1.0) / 2.0)
        return numerator / numerator.sum(dim=1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def distances(self, embeddings):
        assignment = self(embeddings)
        labels = assignment.argmax(dim=1)
        distances = torch.linalg.vector_norm(embeddings - self.cluster_centers[labels], dim=1)
        return labels.cpu().numpy(), distances.cpu().numpy(), assignment.max(dim=1).values.cpu().numpy()


class ReDGAE(nn.Module):
    """Source-faithful ReDGAE used by the released GuardFL implementation."""

    def __init__(self, n_features, n_neurons, embedding_size, alpha1, alpha2, activation="ReLU"):
        super().__init__()
        activations = {"relu": F.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}
        try:
            hidden_activation = activations[activation.lower()]
        except KeyError as exc:
            raise ValueError("g2uard_activation must be ReLU, Sigmoid, or Tanh") from exc
        self.gcn_1 = GraphConvSparse(n_features, n_neurons, hidden_activation)
        self.gcn_2 = GraphConvSparse(n_neurons, embedding_size, lambda value: value)
        self.embedding_size = embedding_size
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.assignment = None

    def reset_assignment(self, n_clusters, device):
        self.assignment = ClusterAssignment(n_clusters, self.embedding_size, self.alpha1).to(device)

    def encode(self, features, adjacency):
        return self.gcn_2(self.gcn_1(features, adjacency), adjacency)

    @staticmethod
    def decode(embeddings):
        return torch.sigmoid(embeddings @ embeddings.t())

    @staticmethod
    def _optimizer(parameters, name, learning_rate):
        if name.lower() == "adam":
            return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.001)
        if name.lower() == "sgd":
            return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
        if name.lower() == "rmsprop":
            return torch.optim.RMSprop(parameters, lr=learning_rate)
        raise ValueError(f"Unsupported g2uard_optimizer: {name}")

    @staticmethod
    def _target_distribution(assignment):
        return F.one_hot(assignment.argmax(dim=1), assignment.shape[1]).float()

    @staticmethod
    def _unconflicted(embeddings, centers, beta1, beta2):
        distance = np.sum((embeddings[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        q = (1.0 / (1.0 + distance))
        q = q / q.sum(axis=1, keepdims=True)
        ordered = np.sort(q, axis=1)
        first = ordered[:, -1]
        second = ordered[:, -2] if q.shape[1] > 1 else np.zeros_like(first)
        keep = (first > beta1) & ((first - second > beta2) if q.shape[1] > 1 else True)
        return np.flatnonzero(keep)

    def _loss(self, assignment, target, reconstructed, adjacency_label):
        reconstruction = F.binary_cross_entropy(reconstructed.reshape(-1), adjacency_label.clamp(1e-6, 1 - 1e-6).reshape(-1))
        clustering = F.kl_div(assignment.clamp_min(1e-12).log(), target, reduction="sum")
        return reconstruction + self.alpha2 * clustering

    def _update_graph(self, adjacency, embeddings, unconflicted):
        """Official confidence-driven graph rewiring step."""
        if len(unconflicted) == 0:
            return adjacency, adjacency + torch.eye(adjacency.shape[0], device=adjacency.device)
        labels = self.assignment(embeddings).argmax(dim=1)
        chosen = torch.as_tensor(unconflicted, device=adjacency.device, dtype=torch.long)
        chosen_embeddings = embeddings[chosen]
        nearest = torch.cdist(self.assignment.cluster_centers, chosen_embeddings).argmin(dim=1)
        representatives = chosen[nearest][labels]
        positive_edges = adjacency[adjacency > 0]
        low = torch.quantile(positive_edges, .01) if positive_edges.numel() else torch.tensor(0.0, device=adjacency.device)
        high = torch.quantile(positive_edges, .99) if positive_edges.numel() else torch.tensor(1.0, device=adjacency.device)
        unconflicted_set = set(int(index) for index in unconflicted)
        for index in unconflicted:
            strong_neighbours = torch.where((adjacency[index] > high) & (torch.arange(adjacency.shape[0], device=adjacency.device) != index))[0]
            representative = int(representatives[index])
            if representative != index and labels[index] == labels[representative] and representative not in strong_neighbours.tolist():
                adjacency[index, representative] = random.uniform(float(high), 1.0)
            for neighbour in strong_neighbours.tolist():
                if neighbour in unconflicted_set and labels[index] != labels[neighbour]:
                    adjacency[index, neighbour] = random.uniform(0.0, float(low))
        return adjacency, adjacency + torch.eye(adjacency.shape[0], device=adjacency.device)

    def pretrain(self, adjacency, features, adjacency_label, optimizer_name, epochs, learning_rate):
        optimizer = self._optimizer(self.parameters(), optimizer_name, learning_rate)
        best_loss, best_embeddings = float("inf"), None
        for _ in range(epochs):
            embeddings = self.encode(features, adjacency)
            loss = F.binary_cross_entropy(self.decode(embeddings).reshape(-1), adjacency_label.clamp(1e-6, 1 - 1e-6).reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if float(loss.detach()) < best_loss:
                best_loss, best_embeddings = float(loss.detach()), embeddings.detach().clone()
        return best_embeddings

    def refine(self, normalized_adjacency, features, adjacency, adjacency_label, optimizer_name, epochs, learning_rate, beta1, beta2):
        optimizer = self._optimizer(self.parameters(), optimizer_name, learning_rate)
        previous_unconflicted = np.asarray([], dtype=int)
        stable_epochs = 0
        best_loss, best_embeddings = float("inf"), None
        target = None
        for epoch in range(epochs):
            embeddings = self.encode(features, normalized_adjacency)
            assignment = self.assignment(embeddings)
            if epoch % 15 == 0:
                target = self._target_distribution(assignment.detach())
                unconflicted = self._unconflicted(
                    embeddings.detach().cpu().numpy(), self.assignment.cluster_centers.detach().cpu().numpy(), beta1, beta2
                )
                if len(previous_unconflicted) < len(unconflicted):
                    previous_unconflicted = unconflicted
                else:
                    stable_epochs += 1
                if epoch == 0 and len(unconflicted):
                    adjacency, adjacency_label = self._update_graph(adjacency, embeddings.detach(), unconflicted)
                if epoch % 20 == 0 and epoch <= 120 and len(unconflicted):
                    adjacency, adjacency_label = self._update_graph(adjacency, embeddings.detach(), unconflicted)
            if stable_epochs >= 15:
                stable_epochs, beta1, beta2 = 0, beta1 * 0.95, beta2 * 0.85
            indices = torch.as_tensor(previous_unconflicted, device=features.device, dtype=torch.long)
            loss = self._loss(assignment[indices], target[indices], self.decode(embeddings), adjacency_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch == 0 or (epoch + 1) % 50 == 0) and float(loss.detach()) < best_loss:
                best_loss, best_embeddings = float(loss.detach()), embeddings.detach().clone()
        return self.assignment.distances(best_embeddings)


class G2uardFL_Server(BasicServer):
    """Adapter of the released GuardFL implementation for the Mirage runner."""

    def __init__(self, params, dataloader):
        super().__init__(params, dataloader)
        self.total_clients = int(params["no_of_total_participants"])
        self.param_keys = [name for name, _ in self.global_model.named_parameters()]
        self.kappa1 = float(params.get("g2uard_kappa1", 0.2))
        self.kappa2 = float(params.get("g2uard_kappa2", 0.2))
        self.kappa3 = float(params.get("g2uard_kappa3", 0.1))
        self.kappa4 = float(params.get("g2uard_kappa4", 0.1))
        self.optimizer_name = str(params.get("g2uard_optimizer", "Adam"))
        self.epochs_pretrain = int(params.get("g2uard_epochs_pretrain", 100))
        self.epochs_cluster = int(params.get("g2uard_epochs_cluster", 150))
        self.lr_pretrain = float(params.get("g2uard_lr_pretrain", 0.01))
        self.lr_cluster = float(params.get("g2uard_lr_cluster", 0.01))
        self.beta1 = float(params.get("g2uard_beta1", 0.5))
        self.beta2 = float(params.get("g2uard_beta2", 0.1))
        self.alpha1 = float(params.get("g2uard_alpha1", 1.0))
        self.alpha2 = float(params.get("g2uard_alpha2", 0.1))
        self.activation = str(params.get("g2uard_activation", "ReLU"))
        self.features = None
        self.adjacency = np.eye(self.total_clients, dtype=np.float32)
        self.previous_global = None
        self.previous_clients = {}
        self.benign_scores = np.random.randn(self.total_clients).astype(np.float32) * 1e-3
        self.median_grad_norm = 0.0
        self.median_model_norm = 0.0
        self.rounds = 0
        self.gae = None
        self.cluster_count = None
        # A diagnostic hand-off for derived defenses. GuardFL itself still
        # aggregates exactly as before; these are the temporary per-round
        # cluster aggregates used internally by its published equation.
        self.last_guardfl_surrogates = None

    def broadcast_upload(self, iteration, benign_client, malicious_client, **kwargs):
        logger.info("Training on global iteration %s ", iteration)
        selected, attackers = self.select_clients(iteration)
        global_copy = self.create_global_model_copy()
        local_models, update_norms = [], []
        for client_id in tqdm(selected):
            client = malicious_client if client_id in attackers else benign_client
            local_model = copy.deepcopy(self.global_model)
            for parameter in local_model.parameters():
                parameter.requires_grad = True
            updated = client.local_train(iteration, local_model, self.train_dataloader[client_id], client_id, test_loader=self.test_dataloader)
            local_models.append(updated)
            update_norms.append(round(model_dist_norm_var(updated, global_copy).item(), 6))
        for client_id, norm in zip(selected, update_norms):
            logger.info("Client %s update norm: %s", client_id, norm)

        global_params = self._params(self.global_model)
        finite_positions = [index for index, model in enumerate(local_models) if self._model_is_finite(model)]
        nonfinite_clients = [selected[index] for index in range(len(selected)) if index not in finite_positions]
        if nonfinite_clients:
            logger.warning(
                "GuardFL rejected non-finite client updates before clustering: %s",
                nonfinite_clients,
            )

        finite_selected = [selected[index] for index in finite_positions]
        finite_params = [self._params(local_models[index]) for index in finite_positions]
        if not finite_selected:
            final_params, accepted, suspicious = global_params, [], []
            self.last_guardfl_surrogates = {
                "good_params": copy.deepcopy(global_params),
                "suspicious_params": None,
                "accepted_clients": [],
                "suspicious_clients": list(nonfinite_clients),
                "iteration": int(iteration),
            }
        elif len(finite_selected) < 3:
            # HDBSCAN/G2 clustering is not meaningful with fewer than three
            # finite updates. Average the finite models, but never admit a
            # non-finite update into the exposed learner.
            norms = [np.linalg.norm(_vec(params) - _vec(global_params)) for params in finite_params]
            final_params, _ = self._aggregate(finite_params, global_params, None, norms)
            accepted, suspicious = list(finite_selected), []
            self.last_guardfl_surrogates = {
                "good_params": copy.deepcopy(final_params),
                "suspicious_params": None,
                "accepted_clients": list(accepted),
                "suspicious_clients": [],
                "iteration": int(iteration),
            }
            logger.warning("GuardFL used finite-update fallback with %d client(s)", len(finite_selected))
        else:
            final_params, accepted, suspicious = self._official_exec(
                finite_selected, finite_params, global_params, iteration
            )
        final_delta = self._params_to_delta(final_params, global_params)
        weights = [1 if client_id in accepted else 0 for client_id in selected]
        logger.info("GuardFL accepted=%s suspicious=%s", accepted, suspicious)
        client_deltas = [
            self._model_delta(model) if position in finite_positions else self._zero_model_delta()
            for position, model in enumerate(local_models)
        ]
        return final_delta, client_deltas, weights

    def aggregation(self, weight_accumulator, aggregated_model_id):
        for name, value in self.global_model.state_dict().items():
            if name in weight_accumulator and torch.is_floating_point(value):
                value.add_(weight_accumulator[name].to(value.device, value.dtype))

    def _params(self, model):
        return [parameter.detach().cpu().numpy().copy() for parameter in model.parameters()]

    def _model_delta(self, model):
        state, global_state = model.state_dict(), self.global_model.state_dict()
        return {name: (state[name].detach().cpu() - global_state[name].detach().cpu()) if torch.is_floating_point(value) else torch.zeros_like(value) for name, value in global_state.items()}

    def _zero_model_delta(self):
        return {
            name: torch.zeros_like(value, device="cpu")
            for name, value in self.global_model.state_dict().items()
        }

    @staticmethod
    def _model_is_finite(model):
        return all(
            bool(torch.isfinite(value).all())
            for value in model.state_dict().values()
            if torch.is_floating_point(value)
        )

    def _params_to_delta(self, final_params, global_params):
        result = self.create_weight_accumulator()
        for name, final, base in zip(self.param_keys, final_params, global_params):
            result[name] = torch.from_numpy(final - base).to(self.params["run_device"])
        return result

    @staticmethod
    def _layer_stats(params):
        output = []
        for operation in range(10):
            values = []
            for parameter in params:
                flat = parameter.reshape(-1)
                if operation == 0: value = np.linalg.norm(flat)
                elif operation == 1: value = np.min(flat)
                elif operation == 2: value = np.max(flat)
                elif operation == 3: value = np.mean(flat)
                elif operation == 4: value = np.std(flat)
                elif operation == 5: value = np.sum(flat)
                elif operation == 6: value = np.median(flat)
                elif operation == 7: value = np.quantile(flat, .05)
                elif operation == 8: value = np.quantile(flat, .95)
                else: value = np.sum(flat > np.mean(flat))
                values.append(value)
            output.extend(values)
        return np.asarray(output, dtype=np.float32)

    def _zscore(self, values, selected):
        mean, std = values[selected].mean(0, keepdims=True), values[selected].std(0, keepdims=True)
        values[selected] = (values[selected] - mean) / (std + 1e-6)
        return values

    def _feature_matrix(self, selected, client_params, global_params):
        layers = len(global_params)
        model_features = np.zeros((self.total_clients, 21), dtype=np.float32)
        parameter_features = np.zeros((self.total_clients, layers * 21), dtype=np.float32)
        temporal_features = np.zeros((self.total_clients, layers * 11), dtype=np.float32)
        global_vector = _vec(global_params)
        for client_id, local in zip(selected, client_params):
            local_vector = _vec(local)
            model_features[client_id] = np.concatenate(([_cos(local_vector, global_vector)], self._layer_stats([local_vector]), self._layer_stats([local_vector - global_vector])))
            parameter_features[client_id] = np.concatenate((self._layer_stats(local), self._layer_stats([a - b for a, b in zip(local, global_params)]), [_cos(a.reshape(-1), b.reshape(-1)) for a, b in zip(local, global_params)]))
            previous = self.previous_clients.get(client_id, global_params)
            previous_global = self.previous_global if self.previous_global is not None else global_params
            current_delta = [a - b for a, b in zip(local, global_params)]
            previous_delta = [a - b for a, b in zip(previous, previous_global)]
            temporal_features[client_id] = np.concatenate((self._layer_stats([a - b for a, b in zip(current_delta, previous_delta)]), [_cos(a.reshape(-1), b.reshape(-1)) for a, b in zip(current_delta, previous_delta)]))
        current = np.concatenate((self._zscore(model_features, selected), self._zscore(parameter_features, selected), self._zscore(temporal_features, selected)), axis=1)
        if self.features is None:
            self.features = np.random.randn(*current.shape).astype(np.float32) * 1e-4
        self.features[selected] = self.kappa1 * self.features[selected] + (1.0 - self.kappa1) * current[selected]

    def _relation(self, selected, vectors, inverse=False):
        output = np.zeros_like(self.adjacency)
        pairs, values = [], []
        for left in range(len(selected)):
            for right in range(left + 1, len(selected)):
                value = vectors(left, right)
                pairs.append((selected[left], selected[right])); values.append(value)
        if not values:
            return output
        values = np.asarray(values); centered = values - values.mean()
        base = 1e6 if values.max() < 5 else (math.e if values.max() > 10 else 1e2)
        scaled = np.power(base, centered)
        scaled = (scaled.max() - scaled if inverse else scaled - scaled.min()) / (scaled.max() - scaled.min() + 1e-6)
        for (left, right), value in zip(pairs, scaled): output[left, right] = output[right, left] = value
        np.fill_diagonal(output, 1.0)
        return output

    def _update_adjacency(self, selected, client_params, global_params):
        local = [_vec(params) for params in client_params]; global_vector = _vec(global_params); gradients = [value - global_vector for value in local]
        cosine = self._relation(selected, lambda i, j: (_cos(local[i], local[j]) + 1) / 2)
        gradient_cosine = self._relation(selected, lambda i, j: (_cos(gradients[i], gradients[j]) + 1) / 2)
        norm = self._relation(selected, lambda i, j: abs(np.linalg.norm(local[i]) - np.linalg.norm(local[j])), inverse=True)
        gradient_norm = self._relation(selected, lambda i, j: abs(np.linalg.norm(gradients[i]) - np.linalg.norm(gradients[j])), inverse=True)
        candidate = (cosine + gradient_cosine + norm + gradient_norm) / 4
        ids = np.ix_(selected, selected)
        if candidate[ids].std() < 1e-2: candidate[ids] = 0
        self.adjacency[ids] = np.clip(self.kappa2 * self.adjacency[ids] + (1 - self.kappa2) * candidate[ids], 0, 1)

    @staticmethod
    def _normalise_adjacency(adjacency, device):
        adjacency = adjacency - torch.diag_embed(torch.diag(adjacency))
        label = adjacency + torch.eye(adjacency.shape[0], device=device)
        degree = (adjacency + torch.eye(adjacency.shape[0], device=device)).sum(1).clamp_min(1e-12)
        normalised = torch.rsqrt(degree)[:, None] * (adjacency + torch.eye(adjacency.shape[0], device=device)) * torch.rsqrt(degree)[None, :]
        return adjacency, label, normalised

    def _cluster(self, selected, iteration):
        features = self.features
        if features.shape[0] >= 128 and features.shape[1] >= 128: from sklearn.decomposition import PCA; features = PCA(128, svd_solver="full", random_state=1027).fit_transform(features)
        elif features.shape[0] >= 64 and features.shape[1] >= 64: from sklearn.decomposition import PCA; features = PCA(64, svd_solver="full", random_state=1027).fit_transform(features)
        elif features.shape[0] >= 32 and features.shape[1] >= 32: from sklearn.decomposition import PCA; features = PCA(32, svd_solver="full", random_state=1027).fit_transform(features)
        chosen = np.ascontiguousarray(features[selected], dtype=np.float32)
        adjacency = np.ascontiguousarray(self.adjacency[np.ix_(selected, selected)], dtype=np.float32)
        import hdbscan
        labels = hdbscan.HDBSCAN(2, min_samples=1).fit_predict(chosen)
        n_clusters = max(2, min(max(2, int(labels.max() + 1)), len(selected) // 2 - 1))
        device = self.params["run_device"]
        if self.gae is None:
            self.gae = ReDGAE(
                chosen.shape[1], max(1, chosen.shape[1] // 4), max(1, chosen.shape[1] // 8),
                self.alpha1, self.alpha2, self.activation,
            ).to(device)
        if self.cluster_count != n_clusters:
            self.cluster_count = n_clusters; self.gae.reset_assignment(n_clusters, device)
        final = None
        for _ in range(2):
            x = torch.from_numpy(chosen).to(device) + torch.randn_like(torch.from_numpy(chosen).to(device)) * 1e-4
            a = torch.from_numpy(adjacency).to(device) + torch.randn_like(torch.from_numpy(adjacency).to(device)) * 1e-4
            raw, target, normalised = self._normalise_adjacency(a, device)
            embeddings = self.gae.pretrain(normalised, x, target, self.optimizer_name, self.epochs_pretrain, self.lr_pretrain)
            from sklearn.cluster import KMeans
            centers = torch.from_numpy(KMeans(n_clusters=n_clusters, n_init=1314, random_state=int(self.params.get("seed", 0)) + iteration).fit(embeddings.cpu().numpy()).cluster_centers_).float().to(device)
            self.gae.assignment.cluster_centers.data.copy_(centers)
            final = self.gae.refine(normalised, x, raw, target, self.optimizer_name, self.epochs_cluster, self.lr_cluster, self.beta1, self.beta2)
            if len(set(final[0].tolist())) >= 2: break
        return final

    def _official_exec(self, selected, client_params, global_params, iteration):
        self._feature_matrix(selected, client_params, global_params)
        self._update_adjacency(selected, client_params, global_params)
        labels, distances, probabilities = self._cluster(selected, iteration)
        self.previous_clients = {client_id: copy.deepcopy(params) for client_id, params in zip(selected, client_params)}
        self.previous_global = copy.deepcopy(global_params); self.rounds += 1
        groups = collections.defaultdict(list)
        for client_id, label in zip(selected, labels): groups[int(label)].append(client_id)
        stats = {label: self.kappa4 * len(ids) / len(selected) + float(self.benign_scores[ids].mean()) for label, ids in groups.items()}
        benign_label = max(stats, key=stats.get)
        min_cluster_score = min(stats.values())
        logger.info(
            "GuardFL clusters: %s",
            {
                label: {
                    "clients": members,
                    "size": len(members),
                    "benign_score": round(stats[label], 6),
                }
                for label, members in sorted(groups.items())
            },
        )
        original_norms = np.array([np.linalg.norm(_vec(params) - _vec(global_params)) for params in client_params])
        model_norms = np.array([np.linalg.norm(_vec(params)) for params in client_params])
        self.median_grad_norm = ((self.rounds - 1) * self.median_grad_norm + np.median(original_norms)) / self.rounds
        self.median_model_norm = ((self.rounds - 1) * self.median_model_norm + np.median(model_norms)) / self.rounds
        clipped = [[base + (value - base) * min(1.0, self.median_grad_norm / (norm + 1e-12)) for base, value in zip(global_params, params)] for params, norm in zip(client_params, original_norms)]
        clipped_norms = np.array([np.linalg.norm(_vec(params) - _vec(global_params)) for params in clipped])
        index = {client_id: idx for idx, client_id in enumerate(selected)}
        benign_group = groups[benign_label]; local_distances = np.array([distances[index[c]] for c in benign_group]); local_scores = self.benign_scores[benign_group]
        keep = (local_distances <= np.quantile(local_distances, .75)) & (local_scores >= np.quantile(local_scores, .25))
        accepted = np.asarray(benign_group)[keep].tolist()
        if not accepted: accepted = [benign_group[int(local_distances.argmin())]]
        trusted_cluster_rejected = [client_id for client_id in benign_group if client_id not in set(accepted)]
        suspicious = [client_id for label, members in groups.items() if stats[label] == min_cluster_score for client_id in members]
        logger.info(
            "GuardFL filtering: trusted_cluster=%s, accepted=%s, trusted_cluster_rejected=%s, lowest_score_cluster=%s",
            benign_label,
            accepted,
            trusted_cluster_rejected,
            suspicious,
        )
        normal, accepted_weights = self._aggregate([clipped[index[c]] for c in accepted], global_params, [distances[index[c]] for c in accepted], [clipped_norms[index[c]] for c in accepted])
        poisoned, _ = self._aggregate([clipped[index[c]] for c in suspicious], global_params, None, [clipped_norms[index[c]] for c in suspicious]) if suspicious else (None, [])
        self.last_guardfl_surrogates = {
            "good_params": copy.deepcopy(normal),
            "suspicious_params": copy.deepcopy(poisoned),
            "accepted_clients": list(accepted),
            "suspicious_clients": list(suspicious),
            "iteration": int(iteration),
        }
        final = normal
        if poisoned is not None:
            normal_norm, poison_norm = np.linalg.norm(_vec(normal) - _vec(global_params)), np.linalg.norm(_vec(poisoned) - _vec(global_params))
            ratio = min(1.0, normal_norm / (poison_norm + 1e-12))
            malicious_score = sum(abs(self.benign_scores[c]) for c in suspicious); total_score = malicious_score + sum(abs(self.benign_scores[c]) for c in accepted)
            weight = 1e-2 * (1.0 + malicious_score / (total_score + 1e-6)) * math.log(1 + self.median_model_norm)
            final = [good + weight * (good - (base + ratio * (bad - base))) for base, good, bad in zip(global_params, normal, poisoned)]
        min_norm = min(clipped_norms[index[c]] for c in accepted); final_norm = np.linalg.norm(_vec(final) - _vec(global_params))
        if final_norm > min_norm: final = [base + (value - base) * min_norm / (final_norm + 1e-12) for base, value in zip(global_params, final)]
        accepted_set, suspicious_set = set(accepted), set(suspicious)
        accepted_weight_by_client = dict(zip(accepted, accepted_weights))
        for client_id in selected:
            if client_id in accepted_set: self.benign_scores[client_id] += self.kappa3 * abs(self.benign_scores[client_id]) * accepted_weight_by_client[client_id] * probabilities[index[client_id]]
            elif client_id in suspicious_set: self.benign_scores[client_id] -= self.kappa3 * abs(self.benign_scores[client_id])
        self.benign_scores = np.tanh(self.benign_scores)
        return final, accepted, suspicious

    @staticmethod
    def _aggregate(models, global_params, distances, norms):
        if not models: return None, []
        weights = np.ones(len(models)) if distances is None else (np.max(distances) - np.asarray(distances)) / (np.max(distances) - np.min(distances) + 1e-6)
        median = np.median(norms); weights *= np.minimum(1.0, median / (np.asarray(norms) + 1e-12))
        aggregate = [sum((base + (model[layer] - base) * weights[idx]) / len(models) for idx, model in enumerate(models)) for layer, base in enumerate(global_params)]
        return aggregate, weights.tolist()
