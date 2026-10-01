import copy
import logging

import torch
import torch.nn.functional as F

from participants.clients.BasicClient import BasicClient
from utils.utils import poisoned_batch_injection

logger = logging.getLogger("logger")


class BadNetClient(BasicClient):
    def __init__(self, params, train_dataloader, test_dataloader):
        super(BadNetClient, self).__init__(params, train_dataloader, test_dataloader)
        self.init_trigger_mask()
        self._init_badnet_trigger()

    def _init_badnet_trigger(self):
        trigger_value = self.params.get("badnet_trigger_value", 1.0)
        trigger_color = self.params.get("badnet_trigger_color")
        shared_trigger = self.params.get("badnet_shared_trigger", True)

        for attacker_id in range(self.params["no_of_adversaries"]):
            self.trigger_set[attacker_id] = self._make_badnet_trigger(
                self.trigger_set[attacker_id],
                trigger_value,
                trigger_color,
            )

        if shared_trigger:
            trigger = self.trigger_set[0].detach().clone()
            mask = self.mask_set[0].detach().clone()
            self.trigger_set = [
                trigger.clone().to(self.params["run_device"])
                for _ in range(self.params["no_of_adversaries"])
            ]
            self.mask_set = [
                mask.clone().to(self.params["run_device"])
                for _ in range(self.params["no_of_adversaries"])
            ]

    def _make_badnet_trigger(self, trigger_template, trigger_value, trigger_color):
        trigger = torch.ones_like(trigger_template, device=self.params["run_device"]) * trigger_value
        if trigger_color is None:
            return trigger

        color = torch.tensor(trigger_color, device=self.params["run_device"], dtype=trigger.dtype)
        if trigger.dim() < 3:
            raise ValueError(
                "badnet_trigger_color requires image-like triggers with channel, height, width dimensions"
            )

        if color.numel() == trigger.shape[0]:
            return color.view(-1, 1, 1).expand_as(trigger).clone()

        if trigger.shape[0] == 1 and color.numel() == 3:
            # RGB colors cannot be represented directly on grayscale datasets.
            # Convert them to luminance so CIFAR-style YAMLs also run on MNIST.
            rgb_to_gray = torch.tensor([0.299, 0.587, 0.114], device=color.device, dtype=color.dtype)
            gray = torch.sum(color * rgb_to_gray).view(1, 1, 1)
            return gray.expand_as(trigger).clone()

        raise ValueError(
            "badnet_trigger_color must match the input channel count, "
            f"got {color.numel()} values for {trigger.shape[0]} channels"
        )

    def _target_label(self, client_id):
        if "badnet_target_label" in self.params:
            return self.params["badnet_target_label"]
        return self.params["poison_label_swap"][client_id]

    def local_train(self, iteration, model, train_loader, client_id, **kwargs):
        cache_model = copy.deepcopy(model)
        cache_model.to(self.params["run_device"])
        cache_model.train()

        trigger_id = 0 if self.params.get("badnet_shared_trigger", True) else client_id
        target_label = self._target_label(client_id)

        optimizer = torch.optim.SGD(
            cache_model.parameters(),
            lr=self.params["poisoned_lr"],
            momentum=self.params["poisoned_momentum"],
            weight_decay=self.params["poisoned_weight_decay"],
        )

        for _ in range(self.params["poisoned_retrain_no_times"]):
            for batch in train_loader:
                inputs, labels = poisoned_batch_injection(
                    batch,
                    trigger=self.trigger_set[trigger_id],
                    mask=self.mask_set[trigger_id],
                    is_eval=False,
                    label_swap=target_label,
                )
                inputs = inputs.to(self.params["run_device"])
                labels = labels.to(self.params["run_device"])
                outputs = cache_model(inputs)
                loss = F.cross_entropy(outputs, labels, label_smoothing=0.001)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        cache_model.eval()
        return cache_model
