import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from participants.servers.BasicServer import BasicServer
from participants.servers.indicator_utils import parameter_distance, validate_indicator_schema, validate_coefficient
from utils.utils import model_dist_norm_var, update_weight_accumulator

logger = logging.getLogger("logger")


class BackdoorIndicator_Server(BasicServer):
    def __init__(self, params, dataloader):
        validate_indicator_schema(params, 'backdoor_indicator')
        validate_coefficient(params.get('indicator_l2_lambda', 0.0))
        super(BackdoorIndicator_Server, self).__init__(params, dataloader)
        self.indicator_loader = self._build_indicator_loader()
        self.indicator_bn_stats = None
        self.main_bn_stats = None

    def _dataset_stats(self):
        if self.params["dataset"].upper() in ["CIFAR10", "CIFAR100"]:
            channels = 3
            mean = torch.tensor([0.4914, 0.4822, 0.4465], device=self.params["run_device"]).view(1, 3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010], device=self.params["run_device"]).view(1, 3, 1, 1)
        elif self.params["dataset"].upper() == "GTSRB":
            channels = 3
            mean = torch.tensor([0.5, 0.5, 0.5], device=self.params["run_device"]).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], device=self.params["run_device"]).view(1, 3, 1, 1)
        elif self.params["dataset"].upper() in ["EMNIST", "MNIST", "FASHIONMNIST", "FASHION_MNIST"]:
            channels = 1
            mean = torch.tensor([0.0], device=self.params["run_device"]).view(1, 1, 1, 1)
            std = torch.tensor([1.0], device=self.params["run_device"]).view(1, 1, 1, 1)
        else:
            raise NotImplementedError
        return channels, mean, std

    def _build_indicator_loader(self):
        channels, mean, std = self._dataset_stats()
        size = self.params.get("indicator_dataset_size", 800)
        batch_size = self.params.get("indicator_batch_size", 64)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.params.get("indicator_seed", 2024))

        kind = self.params.get("indicator_ood_type", "noise")
        if kind == "paper":
            raw_inputs = self._load_paper_ood_inputs(size, channels, generator)
        elif kind == "noise":
            raw_inputs = torch.rand(size, channels, 32, 32, generator=generator)
        elif kind == "binary_mask":
            raw_inputs = (torch.rand(size, channels, 32, 32, generator=generator) > 0.5).float()
        else:
            raise NotImplementedError(f"Unsupported indicator_ood_type: {kind}")

        inputs = (raw_inputs.to(self.params["run_device"]) - mean) / std
        labels = torch.arange(size, dtype=torch.long, device=self.params["run_device"]) % self.params["class_num"]
        dataset = TensorDataset(inputs.detach().cpu(), labels.detach().cpu())
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    def _load_paper_ood_inputs(self, size, channels, generator):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        task = self.params["dataset"].upper()
        download = self.params.get("indicator_download", True)

        if task == "CIFAR10":
            indicator_dataset = datasets.CIFAR100(
                self.params["data_dir"],
                train=True,
                download=download,
                transform=transform,
            )
            dataset_name = "CIFAR100"
        else:
            indicator_dataset = datasets.CIFAR10(
                self.params["data_dir"],
                train=True,
                download=download,
                transform=transform,
            )
            dataset_name = "CIFAR10"

        indices = torch.randperm(len(indicator_dataset), generator=generator)[:size]
        raw_inputs = []
        for index in indices:
            image, _ = indicator_dataset[int(index)]
            if channels == 1 and image.size(0) == 3:
                image = image.mean(dim=0, keepdim=True)
            elif channels == 3 and image.size(0) == 1:
                image = image.repeat(3, 1, 1)
            raw_inputs.append(image)

        logger.info(f"BackdoorIndicator paper OOD dataset: {dataset_name}, size={len(raw_inputs)}")
        return torch.stack(raw_inputs, dim=0)

    def _collect_bn_stats(self, model):
        stats = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                stats[name] = (
                    module.running_mean.detach().clone(),
                    module.running_var.detach().clone(),
                    module.num_batches_tracked.detach().clone()
                    if module.num_batches_tracked is not None else None,
                )
        return stats

    def _load_bn_stats(self, model, stats):
        for name, module in model.named_modules():
            if name not in stats:
                continue
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                running_mean, running_var, batches = stats[name]
                module.running_mean.data.copy_(running_mean.to(module.running_mean.device))
                module.running_var.data.copy_(running_var.to(module.running_var.device))
                if batches is not None and module.num_batches_tracked is not None:
                    module.num_batches_tracked.data.copy_(batches.to(module.num_batches_tracked.device))

    def _indicator_regularizer(self, model, reference_model):
        reference = dict(reference_model.named_parameters())
        return parameter_distance(model, reference)

    def _next_indicator_batch(self, iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(self.indicator_loader)
            return next(iterator), iterator

    def _train_indicator_model(self):
        reference_model = copy.deepcopy(self.global_model).to(self.params["run_device"])
        indicator_model = copy.deepcopy(self.global_model).to(self.params["run_device"])
        indicator_model.train()
        optimizer = torch.optim.SGD(
            indicator_model.parameters(),
            lr=self.params.get("indicator_lr", 0.01),
            momentum=self.params.get("indicator_momentum", 0.9),
            weight_decay=self.params.get("indicator_weight_decay", 0.0005),
        )
        lambda_reg = self.params.get("indicator_l2_lambda", 0.0)

        if "indicator_train_iterations" in self.params:
            iterator = iter(self.indicator_loader)
            for _ in range(self.params["indicator_train_iterations"]):
                (inputs, labels), iterator = self._next_indicator_batch(iterator)
                inputs = inputs.to(self.params["run_device"])
                labels = labels.to(self.params["run_device"])
                logits = indicator_model(inputs)
                loss = F.cross_entropy(logits, labels)
                if lambda_reg > 0:
                    loss = loss + lambda_reg * self._indicator_regularizer(indicator_model, reference_model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            for _ in range(self.params.get("indicator_train_epochs", 1)):
                for inputs, labels in self.indicator_loader:
                    inputs = inputs.to(self.params["run_device"])
                    labels = labels.to(self.params["run_device"])
                    logits = indicator_model(inputs)
                    loss = F.cross_entropy(logits, labels)
                    if lambda_reg > 0:
                        loss = loss + lambda_reg * self._indicator_regularizer(indicator_model, reference_model)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        indicator_model.eval()
        return indicator_model

    @torch.no_grad()
    def _indicator_accuracy_array(self, model):
        model.eval()
        correct = torch.zeros(self.params["class_num"], device=self.params["run_device"])
        total = torch.zeros(self.params["class_num"], device=self.params["run_device"])

        for inputs, labels in self.indicator_loader:
            inputs = inputs.to(self.params["run_device"])
            labels = labels.to(self.params["run_device"])
            preds = model(inputs).argmax(dim=1)
            for class_idx in range(self.params["class_num"]):
                mask = labels == class_idx
                if mask.sum() == 0:
                    continue
                total[class_idx] += mask.sum()
                correct[class_idx] += (preds[mask] == labels[mask]).sum()

        return torch.where(total > 0, correct / total.clamp(min=1), torch.zeros_like(total))

    def pre_process(self, *args, **kwargs):
        self.main_bn_stats = self._collect_bn_stats(self.global_model)
        indicator_model = self._train_indicator_model()
        self.indicator_bn_stats = self._collect_bn_stats(indicator_model)
        self._load_bn_stats(indicator_model, self.main_bn_stats)
        self.global_model = indicator_model
        logger.info("BackdoorIndicator injected OOD indicator task into broadcast model")
        return True

    def _inspect_client_model(self, updated_model):
        inspect_model = copy.deepcopy(updated_model).to(self.params["run_device"])
        if self.indicator_bn_stats is not None:
            self._load_bn_stats(inspect_model, self.indicator_bn_stats)
        acc_array = self._indicator_accuracy_array(inspect_model)
        indicator_accuracy = acc_array.max().item()
        return indicator_accuracy, acc_array.detach().cpu().tolist()

    def broadcast_upload(self, iteration, benign_client, malicious_client, **kwargs):
        logger.info(f"Training on global iteration {iteration} ")

        selected_clients_list, malicious_clients_list = self.select_clients(iteration)
        weight_accumulator = self.create_weight_accumulator()
        weight_accumulator_by_client = []
        update_norm_list = []
        indicator_scores = []
        accepted_flags = []
        global_model_copy = self.create_global_model_copy()

        for client_id in tqdm(selected_clients_list):
            if client_id in malicious_clients_list:
                client = malicious_client
            else:
                client = benign_client
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

            indicator_accuracy, acc_array = self._inspect_client_model(updated_model)
            indicator_scores.append(indicator_accuracy)
            threshold = self.params.get("indicator_reject_threshold", 0.95)
            threshold = threshold / 100.0 if threshold > 1 else threshold
            accepted = indicator_accuracy < threshold
            accepted_flags.append(accepted)

            if accepted:
                weight_accumulator, single_wa = update_weight_accumulator(
                    updated_model,
                    copy.deepcopy(self.global_model),
                    weight_accumulator,
                )
            else:
                _, single_wa = update_weight_accumulator(
                    updated_model,
                    copy.deepcopy(self.global_model),
                    self.create_weight_accumulator(),
                )
            weight_accumulator_by_client.append(single_wa)
            del local_model

            logger.info(
                f"Client {client_id} indicator_accuracy={indicator_accuracy:.6f}, "
                f"accepted={accepted}, indicator_acc_array={acc_array}"
            )

        if sum(accepted_flags) == 0:
            logger.info("BackdoorIndicator accepted 0 clients; aggregation will skip this round")

        for client_ind, client_id in enumerate(selected_clients_list):
            logger.info(
                f"Client {client_id} update norm: {update_norm_list[client_ind]}, "
                f"indicator_score={indicator_scores[client_ind]:.6f}, "
                f"accepted={accepted_flags[client_ind]}"
            )

        aggregated_model_id = [1 if accepted else 0 for accepted in accepted_flags]
        return weight_accumulator, weight_accumulator_by_client, aggregated_model_id
