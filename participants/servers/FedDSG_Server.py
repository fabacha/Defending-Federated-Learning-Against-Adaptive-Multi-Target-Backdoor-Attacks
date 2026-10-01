"""FedDSG (Zhang et al., Cybersecurity 2026) for the SATML server API.

Paper: https://doi.org/10.1186/s42400-026-00625-z
Requires a clean, server-held, training-disjoint anchor set. The SATML data
loader reserves it before client partitioning when defense_method=feddsg.
"""

import copy
import logging
import math

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.nn import functional as F
from tqdm import tqdm

from participants.servers.BasicServer import BasicServer

logger = logging.getLogger("logger")


def semantic_bias_score(delta_bias, steps=1, eps=1e-12):
    """SBF, paper Eqs. (4)-(11): return score and dominant class."""
    bias = np.asarray(delta_bias, dtype=np.float64).reshape(-1) / max(1, steps)
    if bias.size < 2 or not np.isfinite(bias).all():
        raise ValueError("FedDSG needs a finite, multi-class classifier bias update")
    gamma = int(np.argmax(bias))
    center = float(np.median(np.delete(bias, gamma)))
    energy = np.square(bias - center)
    peak = float(energy[gamma])
    total = float(energy.sum())
    concentration = peak / (total + eps)
    strength = peak / (total - peak + eps)
    return concentration * math.log1p(strength), gamma


def semantic_bias_filter(bias_updates, steps):
    """SBF, paper Eq. (12): 1D k-means, vote, remove matching class."""
    n = len(bias_updates)
    scores, classes = zip(*(semantic_bias_score(v, s) for v, s in zip(bias_updates, steps)))
    scores = np.asarray(scores, dtype=np.float64)
    classes = np.asarray(classes, dtype=int)
    if n < 2 or np.ptp(scores) <= 1e-12:
        # No meaningful 2-cluster split. The paper does not define this case.
        return set(), scores, classes, None
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(scores[:, None])
    suspicious = int(np.argmax([scores[labels == c].mean() for c in (0, 1)]))
    votes = np.bincount(classes[labels == suspicious],
                        weights=scores[labels == suspicious], minlength=bias_updates[0].numel())
    if not np.any(votes > 0):
        return set(), scores, classes, None
    target = int(np.argmax(votes))
    return {i for i, cls in enumerate(classes) if cls == target}, scores, classes, target


def constrain_bias(delta_bias, reference, cone_width, eps=1e-12):
    """GDC, paper Eqs. (15)-(17), changing only the bias block."""
    bias = delta_bias.double()
    ref = reference.double()
    ref_squared = torch.dot(ref, ref)
    if float(ref_squared) <= eps:
        return bias.clone()
    parallel = (torch.dot(bias, ref) / (ref_squared + eps)) * ref
    orthogonal = bias - parallel
    orthogonal_norm = torch.linalg.vector_norm(orthogonal)
    limit = cone_width * torch.linalg.vector_norm(parallel)
    if orthogonal_norm > limit:
        orthogonal = orthogonal * (limit / orthogonal_norm)
    return parallel + orthogonal


class FedDSG_Server(BasicServer):
    def __init__(self, params, dataloader):
        super().__init__(params, dataloader)
        self.anchor_loader = getattr(dataloader, "server_anchor_dataloader", None)
        if self.anchor_loader is None:
            raise ValueError("FedDSG needs clean, training-disjoint server_anchor_dataloader")
        self.reference_ema = None
        self.reference_mu = float(params.get("feddsg_reference_mu", 0.1))
        self.cone_width = float(params.get("feddsg_cone_width", 1.0))
        if not (0 < self.reference_mu <= 1) or not (self.cone_width > 0 and math.isfinite(self.cone_width)):
            raise ValueError("Invalid FedDSG reference EMA or cone width")
        heads = [(name, layer) for name, layer in self.global_model.named_modules()
                 if isinstance(layer, torch.nn.Linear)]
        if not heads or heads[-1][1].bias is None:
            raise ValueError("FedDSG needs a final Linear classifier with bias")
        prefix = heads[-1][0] + "." if heads[-1][0] else ""
        self.bias_key = prefix + "bias"
        self.last_feddsg = {}

    def _anchor_reference(self):
        model = self.global_model
        was_training = model.training
        model.eval()
        bias = dict(model.named_parameters())[self.bias_key]
        gradient = torch.zeros_like(bias, dtype=torch.float64)
        count = 0
        for images, labels in self.anchor_loader:
            images = images.to(self.params["run_device"])
            labels = labels.to(self.params["run_device"])
            loss = F.cross_entropy(model(images), labels, reduction="sum")
            contribution = torch.autograd.grad(loss, bias)[0]
            gradient += contribution.detach().double()
            count += len(labels)
        model.train(was_training)
        if count == 0:
            raise ValueError("FedDSG anchor set is empty")
        raw = (-gradient / count).detach().cpu()
        self.reference_ema = (raw if self.reference_ema is None else
                              (1 - self.reference_mu) * self.reference_ema + self.reference_mu * raw)
        return self.reference_ema

    def broadcast_upload(self, iteration, benign_client, malicious_client, **kwargs):
        ids, malicious = self.select_clients(iteration)
        reference = self._anchor_reference()
        base = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
        updates, steps = [], []
        for client_id in tqdm(ids):
            client = malicious_client if client_id in malicious else benign_client
            local = copy.deepcopy(self.global_model).train()
            local.requires_grad_(True)
            trained = client.local_train(iteration, local, self.train_dataloader[client_id],
                                         client_id, test_loader=self.test_dataloader)
            updates.append({k: value.detach().cpu() - base[k]
                            for k, value in trained.state_dict().items()})
            epochs = (self.params["poisoned_retrain_no_times"] if client_id in malicious
                      else self.params["benign_retrain_no_times"])
            steps.append(max(1, len(self.train_dataloader[client_id]) * int(epochs)))
            del local, trained
        rejected_idx, scores, classes, target = semantic_bias_filter(
            [update[self.bias_key] for update in updates], steps)
        accepted = [i for i in range(len(ids)) if i not in rejected_idx]
        accumulator = self.create_weight_accumulator()
        for i in accepted:
            update = updates[i]
            adjusted_bias = constrain_bias(update[self.bias_key], reference, self.cone_width)
            for name, value in update.items():
                change = adjusted_bias if name == self.bias_key else value
                accumulator[name].add_(change.to(accumulator[name]))
        weights = [int(i in accepted) for i in range(len(ids))]
        self.last_feddsg = {"scores": scores.tolist(), "dominant_classes": classes.tolist(),
                            "inferred_target": target, "rejected": [ids[i] for i in rejected_idx],
                            "retained": [ids[i] for i in accepted], "anchor_count": len(self.anchor_loader.dataset)}
        logger.info("FedDSG round=%s %s", iteration, self.last_feddsg)
        return accumulator, updates, weights
