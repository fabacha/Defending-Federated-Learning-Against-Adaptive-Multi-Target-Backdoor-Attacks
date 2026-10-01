import copy

import numpy as np
import torch
import torchvision
from torch import nn

import models.resnet
import models.vgg
import random

from torchvision import transforms
from utils.visualize import visualize_batch, visualize

torch.set_printoptions(precision=2, threshold=32, edgeitems=32, linewidth=140, profile=None)

from utils.utils import poisoned_batch_injection


class BasicClient():
    def __init__(self, params, train_dataloader, test_dataloader, **kwargs):
        self.params = params
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = nn.functional.cross_entropy
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.classifier_name = "linear"
        self.create_model()
        self.init_test_sample_cache()

    def create_model(self):

        if "ResNet" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                local_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="CIFAR")
            elif self.params["dataset"].upper() == "CIFAR100":
                local_model = getattr(models.resnet, self.params["model_type"])(num_classes=100, dataset="CIFAR")
            elif self.params["dataset"].upper() == "GTSRB":
                local_model = getattr(models.resnet, self.params["model_type"])(num_classes=43, dataset="GTSRB")
            elif self.params["dataset"].upper() in ["EMNIST", "MNIST", "FASHIONMNIST", "FASHION_MNIST"]:
                local_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="MNIST")
            self.classifier_name = "linear"

        elif "VGG" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                local_model = getattr(models.vgg, self.params["model_type"])(num_classes=10)
            elif self.params["dataset"].upper() == "CIFAR100":
                local_model = getattr(models.vgg, self.params["model_type"])(num_classes=100)

            self.classifier_name = "classifier"
        elif "MobileNet" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                local_model = torchvision.models.mobilenet_v2(num_classes=10)
            elif self.params["dataset"].upper() == "CIFAR100":
                local_model = torchvision.models.mobilenet_v2(num_classes=100)
            self.classifier_name = "classifier"

        self.local_model = local_model.to(self.params["run_device"])

    def local_train(self, iteration, model, train_loader, client_id):
        raise NotImplementedError

    def local_test_once(self, model, test_dataloader, is_poisoned=False, client_id=0):
        '''
        测试本地模型
        :param test_dataloader:
        :param is_poisoned: 是否投毒环境下
        :param client_id:
        :return: acc, loss
        '''
        model.eval()

        with torch.no_grad():
            total_loss = 0.
            total_correct = 0.
            total_num = 0.
            criterion = nn.CrossEntropyLoss(reduction='sum')
            for i, batch in enumerate(test_dataloader):
                if is_poisoned:
                    # 如果需要测试ASR，则需要对batch进行投毒
                    batch = poisoned_batch_injection(batch, self.trigger_set[client_id], self.mask_set[client_id],
                                                     is_eval=True,
                                                     label_swap=self.params["poison_label_swap"][client_id])
                inputs, labels = batch
                inputs = inputs.to(self.params["run_device"])
                labels = labels.to(self.params["run_device"])
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                pred = outputs.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum().item()
                total_num += labels.size(0)

            acc = total_correct / total_num
            loss = total_loss / total_num
            return acc, loss

    def get_lr(self, iteration):
        lr = self.params["benign_lr"] * (2400 - iteration) / 2400  # original
        if lr <= 0.01:
            lr = 0.01
        # if self.params["lr_method"] == "exp":
        #     lr = self.params['benign_lr'] * (self.params['gamma'] ** iteration)
        # elif self.params["lr_method"] == "linear":
        #     if (iteration > self.params['poisoned_start_iteration'] and iteration < self.params[
        #         'poisoned_end_iteration']) or iteration > 1900:
        #         lr = 0.02
        #     else:
        #         lr_init = self.params['benign_lr']
        #         target_lr = self.params['target_lr']
        #         current_iters = iteration - self.params['start_iteration']
        #         training_iters = self.params['end_iteration'] - self.params['start_iteration']
        #         if current_iters <= training_iters / 2.:
        #             lr = current_iters * (target_lr - lr_init) / (training_iters / 2. - 1) + lr_init - (
        #                     target_lr - lr_init) / (training_iters / 2. - 1)
        #         else:
        #             lr = (current_iters - training_iters / 2) * (-target_lr) / (
        #                     training_iters / 2) + target_lr
        #         if lr <= 0.002:
        #             lr = 0.002
        return lr

        raise NotImplementedError

    def init_trigger_mask(self):
        mask_list = []
        dataloader = self.train_dataloader[1]
        sample_data = next(iter(dataloader))[0][0]
        n_adversaries = self.params["no_of_adversaries"]
        if self.params["poisoned_pattern_choose"] == 1:  # 固定位置触发器，pixel (3,3) , fixed trigger
            self.trigger_set = []
            for i in range(n_adversaries):
                if self.params["malicious_train_algo"] == "A3FL" or self.params["malicious_train_algo"] == "Mirage":
                    trigger = torch.ones(sample_data.shape, device=self.params["run_device"]) * 0.5
                else:
                    trigger = (torch.rand(sample_data.shape) - 0.5) * 2
                # trigger = torch.ones_like(sample_data)
                # trigger[channel, height, width]

                self.trigger_set.append(trigger.to(self.params["run_device"]))
            mask1 = torch.zeros_like(sample_data).to(self.params["run_device"])
            mask2 = torch.zeros_like(sample_data).to(self.params["run_device"])
            mask3 = torch.zeros_like(sample_data).to(self.params["run_device"])
            mask4 = torch.zeros_like(sample_data).to(self.params["run_device"])
            mask5 = torch.zeros_like(sample_data).to(self.params["run_device"])
            trigger_size = self.params["trigger_size"]
            mask1[:, 1:1 + trigger_size, 1:1 + trigger_size] = 1
            mask2[:, -1 - trigger_size:-1, 1:1 + trigger_size] = 1
            mask3[:, -1 - trigger_size:-1, -1 - trigger_size:-1] = 1
            mask4[:, 1:1 + trigger_size, -1 - trigger_size:-1] = 1
            mask5[:, 13:13 + trigger_size, 13:13 + trigger_size] = 1
            self.mask_set = [mask1, mask2, mask3, mask4, mask5]
            test_trigger_mask = [self.trigger_set[i] * self.mask_set[i] for i in range(len(self.trigger_set))]
        elif self.params["poisoned_pattern_choose"] == 2:
            mask_list = []
            self.trigger_set = []
            self.mask_set = []
            dataloader = self.train_dataloader[1]
            sample_data = next(iter(dataloader))[0][0]
            n_adversaries = self.params["no_of_adversaries"]
            for i in range(n_adversaries):
                if self.params["malicious_train_algo"] == "A3FL" or self.params["malicious_train_algo"] == "Mirage":
                    trigger = torch.ones(sample_data.shape, device=self.params["run_device"]) * 0.5
                else:
                    trigger = (torch.rand(sample_data.shape, device=self.params["run_device"]) - 0.5) * 2
                mask = torch.ones(sample_data.shape).to(self.params["run_device"])
                # visualize(trigger, f"trigger_{i}")
                self.trigger_set.append(trigger)
                self.mask_set.append(mask)


        elif self.params["poisoned_pattern_choose"] == 3:  # 固定位置触发器，pixel (3,3) , fixed trigger
            self.trigger_set = []
            for i in range(n_adversaries):
                trigger = (torch.rand(sample_data.shape) - 0.5) * 2
                self.trigger_set.append(trigger.to(self.params["run_device"]))

            mask1 = torch.zeros_like(sample_data).to(self.params["run_device"])
            mask2 = torch.zeros_like(sample_data).to(self.params["run_device"])
            mask3 = torch.zeros_like(sample_data).to(self.params["run_device"])
            mask4 = torch.zeros_like(sample_data).to(self.params["run_device"])
            mask5 = torch.zeros_like(sample_data).to(self.params["run_device"])
            trigger_size = self.params["trigger_size"]
            mask1[:, 1:1 + trigger_size, 1:1 + trigger_size] = 1
            mask2[:, -1 - trigger_size:-1, 1:1 + trigger_size] = 1
            mask3[:, -1 - trigger_size:-1, -1 - trigger_size:-1] = 1
            mask4[:, 1:1 + trigger_size, -1 - trigger_size:-1] = 1
            mask5[:, 13:13 + trigger_size, 13:13 + trigger_size] = 1
            self.mask_set = [mask1, mask2, mask3, mask4, mask5]
            # 设置torch输出全部内容
    def init_test_sample_cache(self):
        samples_per_class_cache = {i: [torch.tensor([]), torch.tensor([])] for i in range(self.params["class_num"])}
        for batch in self.test_dataloader:
            inputs, labels = batch
            for i in range(self.params["class_num"]):
                indices_of_class_i = labels == i
                if indices_of_class_i.sum() > 0 and len(samples_per_class_cache[i][0]) < 200:
                    samples_per_class_cache[i][0] = torch.cat(
                        (samples_per_class_cache[i][0], inputs[indices_of_class_i]),
                        dim=0)
                    samples_per_class_cache[i][1] = torch.cat(
                        (samples_per_class_cache[i][1], labels[indices_of_class_i]),
                        dim=0)
        print(
            {f"class_{i}: {len(samples_per_class_cache[i][0])}" for i in range(self.params["class_num"])})
        self.test_sample_cache = samples_per_class_cache
