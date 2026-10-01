import copy
import logging

import torch
import torch.nn.functional as F

from participants.clients.BasicClient import BasicClient

logger = logging.getLogger("logger")


class A3FLClient(BasicClient):
    def __init__(self, params, train_dataloader, test_dataloader):
        super(A3FLClient, self).__init__(params, train_dataloader, test_dataloader)
        self.init_trigger_mask()
        self._sync_shared_trigger(self.trigger_set[0], self.mask_set[0])

    def _sync_shared_trigger(self, trigger, mask):
        if not self.params.get("a3fl_shared_trigger", True):
            return
        self.trigger_set = [
            trigger.detach().clone().to(self.params["run_device"])
            for _ in range(self.params["no_of_adversaries"])
        ]
        self.mask_set = [
            mask.detach().clone().to(self.params["run_device"])
            for _ in range(self.params["no_of_adversaries"])
        ]

    def _target_label(self, client_id):
        if "a3fl_target_label" in self.params:
            return self.params["a3fl_target_label"]
        return self.params["poison_label_swap"][client_id]

    def _apply_trigger(self, inputs, trigger, mask):
        trigger = trigger.to(inputs.device)
        mask = mask.to(inputs.device)

        if self.params["poisoned_pattern_choose"] == 1:
            return trigger * mask + (1 - mask) * inputs

        if self.params["poisoned_pattern_choose"] == 2:
            alpha = self.params.get("blend_alpha", 0.3)
            return trigger * alpha + (1 - alpha) * inputs

        raise NotImplementedError(f"Unsupported poisoned_pattern_choose: {self.params['poisoned_pattern_choose']}")

    def _poison_batch(self, batch, trigger, mask, target_label):
        inputs, labels = copy.deepcopy(batch)
        inputs = inputs.to(self.params["run_device"])
        labels = labels.to(self.params["run_device"])

        poison_fraction = self.params.get("a3fl_poison_fraction", None)
        if poison_fraction is None:
            poison_count = min(self.params["poisoned_len"], len(labels))
        else:
            poison_count = max(1, int(len(labels) * poison_fraction))

        indices = torch.arange(poison_count, device=inputs.device)
        inputs[indices] = self._apply_trigger(inputs[indices], trigger, mask)
        labels[indices] = target_label
        return inputs, labels

    def _trigger_batch_keep_labels(self, batch, trigger, mask):
        inputs, labels = batch
        inputs = inputs.to(self.params["run_device"])
        labels = labels.to(self.params["run_device"])
        return self._apply_trigger(inputs, trigger, mask), labels

    def _model_cosine_similarity(self, model_a, model_b):
        chunks_a = []
        chunks_b = []
        for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
            chunks_a.append(param_a.detach().view(-1))
            chunks_b.append(param_b.detach().view(-1))
        vec_a = torch.cat(chunks_a)
        vec_b = torch.cat(chunks_b)
        return F.cosine_similarity(vec_a, vec_b, dim=0).clamp(min=0.0).item()

    def _optimize_trigger(self, global_model, train_loader, client_id):
        target_label = self._target_label(client_id)
        mask = self.mask_set[0 if self.params.get("a3fl_shared_trigger", True) else client_id]
        trigger = self.trigger_set[0 if self.params.get("a3fl_shared_trigger", True) else client_id]
        trigger = trigger.detach().clone().to(self.params["run_device"])
        trigger.requires_grad_(True)

        adversarial_model = copy.deepcopy(global_model).to(self.params["run_device"])
        adversarial_model.train()
        adv_optimizer = torch.optim.SGD(
            adversarial_model.parameters(),
            lr=self.params.get("a3fl_adv_lr", 0.01),
            momentum=self.params.get("a3fl_adv_momentum", 0.0),
            weight_decay=self.params.get("a3fl_adv_weight_decay", 0.0),
        )

        trigger_steps = self.params.get("a3fl_trigger_steps", 5)
        outer_steps = self.params.get("a3fl_outer_steps", 5)
        max_batches = self.params.get("a3fl_trigger_batches", 1)
        trigger_lr = self.params.get("a3fl_trigger_lr", 0.01)
        trigger_eps = self.params.get("a3fl_trigger_eps", 2.0)
        lambda_base = self.params.get("a3fl_lambda", 1.0)

        global_model.eval()
        for outer_step in range(outer_steps):
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= max_batches:
                    break

                inputs, labels = batch
                inputs = inputs.to(self.params["run_device"])
                labels = labels.to(self.params["run_device"])
                target_labels = torch.full_like(labels, target_label)

                for _ in range(trigger_steps):
                    triggered_inputs = self._apply_trigger(inputs, trigger, mask)
                    lambda_adapt = lambda_base * self._model_cosine_similarity(adversarial_model, global_model)
                    loss_current = F.cross_entropy(global_model(triggered_inputs), target_labels)
                    loss_adversarial = F.cross_entropy(adversarial_model(triggered_inputs), target_labels)
                    loss = loss_current + lambda_adapt * loss_adversarial

                    if trigger.grad is not None:
                        trigger.grad.zero_()
                    loss.backward()
                    with torch.no_grad():
                        trigger -= trigger_lr * trigger.grad.sign()
                        trigger.clamp_(min=-trigger_eps, max=trigger_eps)
                        trigger.requires_grad_(True)

                triggered_inputs = self._apply_trigger(inputs, trigger.detach(), mask)
                adv_loss = F.cross_entropy(adversarial_model(triggered_inputs), labels)
                adv_optimizer.zero_grad()
                adv_loss.backward()
                adv_optimizer.step()

            logger.info(
                f"A3FL trigger optimization outer_step={outer_step + 1}/{outer_steps}, "
                f"target={target_label}"
            )

        trigger = trigger.detach()
        if self.params.get("a3fl_shared_trigger", True):
            self._sync_shared_trigger(trigger, mask)
        else:
            self.trigger_set[client_id] = trigger
        return trigger, mask

    def local_train(self, iteration, model, train_loader, client_id, **kwargs):
        cache_model = copy.deepcopy(model)
        cache_model.to(self.params["run_device"])

        trigger, mask = self._optimize_trigger(cache_model, train_loader, client_id)
        target_label = self._target_label(client_id)

        optimizer = torch.optim.SGD(
            cache_model.parameters(),
            lr=self.params["poisoned_lr"],
            momentum=self.params["poisoned_momentum"],
            weight_decay=self.params["poisoned_weight_decay"],
        )

        cache_model.train()
        for _ in range(self.params["poisoned_retrain_no_times"]):
            for batch in train_loader:
                inputs, labels = self._poison_batch(batch, trigger, mask, target_label)
                outputs = cache_model(inputs)
                loss = F.cross_entropy(outputs, labels, label_smoothing=0.001)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        cache_model.eval()
        return cache_model
