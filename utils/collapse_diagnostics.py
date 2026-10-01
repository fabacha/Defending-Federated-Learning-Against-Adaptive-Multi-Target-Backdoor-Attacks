import copy
import json
import logging
import math
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("logger")


@contextmanager
def preserve_random_state(loader=None):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    generators = {}
    for owner in (loader, getattr(loader, "sampler", None),
                  getattr(getattr(loader, "batch_sampler", None), "sampler", None)):
        generator = getattr(owner, "generator", None)
        if isinstance(generator, torch.Generator):
            generators[id(generator)] = (generator, generator.get_state())
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for generator, state in generators.values():
            generator.set_state(state)


class CollapseDiagnostics:
    """Read-only fixed-probe evaluation of uploads and aggregation."""

    def __init__(self, params, loader):
        if getattr(loader, "num_workers", 0):
            raise ValueError("Collapse diagnostics require a single-process probe loader to preserve RNG state")
        self.samples = int(params.get("collapse_probe_samples", 256))
        self.batch_size = int(params.get("collapse_probe_batch_size", 64))
        if self.samples < 2 or self.batch_size < 2:
            raise ValueError("Collapse diagnostics require at least two samples per probe and batch")
        images, labels = [], []
        count = 0
        with preserve_random_state(loader):
            for inputs, targets in loader:
                take = min(len(inputs), self.samples - count)
                images.append(inputs[:take].detach().cpu().clone())
                labels.append(targets[:take].detach().cpu().clone())
                count += take
                if count >= self.samples:
                    break
        if count < 2:
            raise ValueError("Collapse diagnostic loader yielded fewer than two examples")
        self.inputs, self.labels = torch.cat(images), torch.cat(labels)
        if not torch.isfinite(self.inputs).all():
            raise ValueError("Collapse diagnostic inputs contain nonfinite values")
        self.records = []

    def _measure(self, model, batch_statistics):
        model.eval()
        batchnorm_count = 0
        if batch_statistics:
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.train()
                    batchnorm_count += 1
        probabilities, targets, loss_sum = [], [], 0.0
        device = next(model.parameters()).device
        count = len(self.labels)
        offset = 0
        with torch.no_grad():
            while offset < count:
                end = min(offset + self.batch_size, count)
                if count - end == 1:
                    end = count
                inputs = self.inputs[offset:end].to(device)
                labels = self.labels[offset:end].to(device)
                logits = model(inputs)
                if not torch.isfinite(logits).all():
                    return {"finite": False, "samples": count, "batch_statistics": batch_statistics}
                loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
                probabilities.append(F.softmax(logits, dim=1).cpu())
                targets.append(labels.cpu())
                offset = end
        if not math.isfinite(loss_sum):
            return {"finite": False, "samples": count, "reason": "nonfinite CE"}
        probabilities, targets = torch.cat(probabilities), torch.cat(targets)
        predictions = probabilities.argmax(1)
        classes = probabilities.shape[1]
        return {
            "finite": True, "samples": count, "accuracy_percent": 100 * predictions.eq(targets).float().mean().item(),
            "ce": loss_sum / count, "mean_confidence": probabilities.max(1).values.mean().item(),
            "prediction_counts": torch.bincount(predictions, minlength=classes).tolist(),
            "label_counts": torch.bincount(targets, minlength=classes).tolist(),
            "batchnorm_layers": batchnorm_count if batch_statistics else None,
        }

    def report(self, model, iteration, stage, client_id=None, role=None):
        with preserve_random_state():
            probe_model = copy.deepcopy(model)
            variances = [module.running_var.detach().flatten().cpu()
                         for module in probe_model.modules()
                         if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.running_var is not None]
            record = {"round": int(iteration), "stage": stage, "client": client_id, "role": role}
            if variances:
                values = torch.cat(variances)
                finite = values[torch.isfinite(values)]
                record["running_variance"] = {
                    "nonfinite_count": int((~torch.isfinite(values)).sum()),
                    "negative_count": int((values < 0).sum()),
                    "min": finite.min().item() if len(finite) else None,
                    "max": finite.max().item() if len(finite) else None,
                }
            record["eval"] = self._measure(probe_model, False)
            record["batch_stats_probe"] = self._measure(probe_model, True)
            del probe_model
        self.records.append(record)
        logger.info("[TALI-COLLAPSE] %s", json.dumps(record, allow_nan=False))
        return record
