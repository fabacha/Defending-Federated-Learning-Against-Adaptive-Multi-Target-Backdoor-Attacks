import torch
import random
import copy
import numpy as np
import torch.utils.data
import logging
from collections import defaultdict

from colorama import Fore
from torchvision import datasets, transforms
from tqdm import tqdm

logger = logging.getLogger("logger")


class MSPDataloader():

    def __init__(self, params):
        self.params = params
        if self.params['load_data_from_pkl'] == True:
            if (str(self.params.get("defense_method", "")).lower() == "feddsg" or
                    "feddsg_anchor_fraction" in self.params):
                raise ValueError("Anchor-matched runs require load_data_from_pkl: false to hold out clean samples before partitioning")
            pre_cached_data = torch.load(self.params['pre_cache_data_path'])
            self.train_dataloader = pre_cached_data['train_dataset']
            self.test_dataloader = pre_cached_data['test_dataset']

        else:
            self.load_dataset()

    def load_dataset(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_digit_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])
        transform_digit_test = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor()
        ])
        transform_grstb = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        transform_grstb_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        if self.params["dataset"].upper() == "CIFAR10":
            self.train_dataset = datasets.CIFAR10(f"{self.params['data_dir']}", train=True, download=True,
                                                  transform=transform_train)
            self.test_dataset = datasets.CIFAR10(f"{self.params['data_dir']}", train=False, download=True,
                                                 transform=transform_test)

        elif self.params["dataset"].upper() == "CIFAR100":
            self.train_dataset = datasets.CIFAR100(f"{self.params['data_dir']}", train=True, download=True,
                                                   transform=transform_train)
            self.test_dataset = datasets.CIFAR100(f"{self.params['data_dir']}", train=False, download=True,
                                                  transform=transform_test)
        elif self.params["dataset"].upper() == "GTSRB":
            # self.train_dataset = datasets.EMNIST(f"{self.params['data_dir']}", train=True, split="mnist", download=True,
            #                                      transform=transform_emnist)
            # self.test_dataset = datasets.EMNIST(f"{self.params['data_dir']}", train=False, split="mnist", transform=transform_emnist)
            self.train_dataset = datasets.GTSRB(f"{self.params['data_dir']}", split="train", download=True,
                                                transform=transform_grstb)
            if self.params.get("clean_data_pipeline", False):
                self.train_dataset.transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5] * 3, [0.5] * 3),
                ])
            else:
                self.train_dataset = [sample for sample in self.train_dataset] * 3
            self.test_dataset = datasets.GTSRB(f"{self.params['data_dir']}", split="test", download=True,
                                               transform=transform_grstb_test)

        else:
            raise NotImplementedError(f"Unsupported dataset: {self.params['dataset']}")

        if (str(self.params.get("defense_method", "")).lower() == "feddsg" or
                "feddsg_anchor_fraction" in self.params):
            eval_transform = (transform_digit_test if self.params["dataset"].upper() in
                              ["EMNIST", "MNIST", "FASHIONMNIST", "FASHION_MNIST"] else
                              transform_grstb_test if self.params["dataset"].upper() == "GTSRB" else
                              transform_test)
            self._reserve_feddsg_anchor(eval_transform)

        indices_per_participant = self.sample_dirichlet_train_data(
            self.params['no_of_total_participants'],
            alpha=self.params['dirichlet_alpha'])
        from torch.utils.data import Subset
        train_loaders = []

        for pos, indices in tqdm(indices_per_participant.items()):
            tmp_subset = Subset(self.train_dataset, indices)
            train_loader = torch.utils.data.DataLoader(
                tmp_subset,
                batch_size=self.params["train_batch_size"],
                shuffle=True,
                drop_last=not self.params.get("clean_data_pipeline", False))
            train_loaders.append(train_loader)

        self.train_dataloader = train_loaders

        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.params["test_batch_size"],
            shuffle=False, drop_last=not self.params.get("clean_data_pipeline", False))

    def _reserve_feddsg_anchor(self, eval_transform):
        """Hold a seeded clean subset out of training before client partitioning."""
        from torch.utils.data import Subset

        fraction = float(self.params.get("feddsg_anchor_fraction", 0.01))
        if not 0 < fraction < 1:
            raise ValueError("feddsg_anchor_fraction must be between 0 and 1")
        source = self.train_dataset
        labels = getattr(source, "targets", None)
        if labels is None:
            labels = [int(label) for _, label in source]
        labels = np.asarray(labels, dtype=int)
        rng = np.random.RandomState(int(self.params.get("feddsg_anchor_seed", self.params["seed"])))
        anchor_indices = []
        for cls in np.unique(labels):
            candidates = np.flatnonzero(labels == cls)
            count = min(len(candidates) - 1, max(1, int(round(len(candidates) * fraction))))
            if count < 1:
                raise ValueError("FedDSG cannot reserve an anchor sample for every class")
            anchor_indices.extend(rng.choice(candidates, count, replace=False).tolist())
        anchor_set = set(anchor_indices)
        training_indices = [i for i in range(len(labels)) if i not in anchor_set]
        anchor_source = copy.copy(source)
        if hasattr(anchor_source, "transform"):
            anchor_source.transform = eval_transform
        self.server_anchor_dataloader = torch.utils.data.DataLoader(
            Subset(anchor_source, sorted(anchor_indices)),
            batch_size=int(self.params.get("feddsg_anchor_batch_size", 128)), shuffle=False)
        self.train_dataset = Subset(source, training_indices)
        logger.info("FedDSG reserved %s clean training samples for server anchor; %s remain for clients",
                    len(anchor_indices), len(training_indices))

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        if self.params.get("clean_data_pipeline", False):
            labels = getattr(self.train_dataset, "targets", None)
            if labels is None:
                labels = [label for _, label in self.train_dataset]
            allocations = {client: [] for client in range(no_participants)}
            labels = np.asarray(labels)
            for label in np.unique(labels):
                indices = np.flatnonzero(labels == label)
                np.random.shuffle(indices)
                counts = np.random.multinomial(len(indices), np.random.dirichlet([alpha] * no_participants))
                for client, shard in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
                    allocations[client].extend(shard.tolist())
            if any(not indices for indices in allocations.values()):
                raise ValueError("Empty clean-FL client shard; increase alpha or reduce client count")
            return allocations
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list
