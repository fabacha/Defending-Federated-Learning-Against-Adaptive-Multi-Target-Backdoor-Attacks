import argparse
import importlib
import logging
import random
import time
from datetime import datetime

import numpy as np
import torch

from datasets.MSP_dataloader import MSPDataloader
from participants.clients.A3FLClient import A3FLClient
from participants.clients.BadNetClient import BadNetClient
from participants.clients.BenignClient import BenignClient
from participants.clients.MirageClient import MirageClient
from participants.servers.BackdoorIndicator_Server import BackdoorIndicator_Server
from participants.servers.G2uardFL_Server import G2uardFL_Server
from participants.servers.Flame_Server import Flame_Server
from participants.servers.No_defense_Server import No_defense_Server
from utils.utils import args_update

logger = logging.getLogger("logger")


class NoAttackClient:
    def __init__(self):
        self.trigger_set = []
        self.mask_set = []
        self.test_sample_cache = {}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optional_class(module_name, class_name):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        logger.debug(f"Optional component unavailable: {module_name}.{class_name} ({exc})")
        return None


def add_optional_defense(registry, names, module_name, class_name):
    cls = optional_class(module_name, class_name)
    if cls is None:
        return
    for name in names:
        registry[name] = cls


def build_defense_registry():
    registry = {
        "nodefense": No_defense_Server,
        "no_defense": No_defense_Server,
        "none": No_defense_Server,
        "backdoor_indicator": BackdoorIndicator_Server,
        "flame": Flame_Server,
        "g2uardfl": G2uardFL_Server,
    }

    optional_defenses = [
        (["fedgrad"], "participants.servers.FedGrad_Server", "FedGrad_Server"),
        (["feddsg"], "participants.servers.FedDSG_Server", "FedDSG_Server"),
        (["pace"], "participants.servers.PACE_Server", "PACE_Server"),
        (["foolsgold"], "participants.servers.FoolsGold_Server", "FoolsGold_Server"),
        (["deepsight"], "participants.servers.DeepSight_Server", "DeepSight_Server"),
        (["multikrum"], "participants.servers.Krum_Server", "MultiKrum_Server"),
        (["oracle"], "participants.servers.Oracle_Server", "Oracle_Server"),
    ]
    for names, module_name, class_name in optional_defenses:
        add_optional_defense(registry, names, module_name, class_name)

    return registry


def create_server(params_loaded, dataloader):
    defense_name = str(params_loaded["defense_method"]).lower().replace("-", "_")
    registry = build_defense_registry()
    if defense_name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise NotImplementedError(
            f"Unknown defense_method: {params_loaded['defense_method']}. "
            f"Available defenses in this checkout: {available}"
        )
    return registry[defense_name](params=params_loaded, dataloader=dataloader)


def create_malicious_client(params_loaded, dataloader):
    if int(params_loaded.get("no_of_adversaries", 0)) <= 0:
        logger.info("No adversaries configured; using clean no-attack client placeholder")
        return NoAttackClient()

    attack_name = str(params_loaded["malicious_train_algo"]).lower()
    if attack_name == "mirage":
        if "mirage_pace_evasion_mode" in params_loaded:
            raise ValueError("Adaptive Mirage variants are not included in this release")
        return MirageClient(params_loaded, dataloader.train_dataloader, dataloader.test_dataloader)
    if attack_name == "a3fl":
        return A3FLClient(params_loaded, dataloader.train_dataloader, dataloader.test_dataloader)
    if attack_name == "badnet":
        return BadNetClient(params_loaded, dataloader.train_dataloader, dataloader.test_dataloader)
    raise ValueError(f"Unsupported attack in this release: {params_loaded['malicious_train_algo']}")


def set_class_num_and_dataset_overrides(params_loaded):
    if params_loaded["dataset"].upper() == "CIFAR10":
        params_loaded["class_num"] = 10
    elif params_loaded["dataset"].upper() == "CIFAR100":
        params_loaded["class_num"] = 100
    elif params_loaded["dataset"].upper() in ["EMNIST", "MNIST", "FASHIONMNIST", "FASHION_MNIST"]:
        params_loaded["class_num"] = 10
    elif params_loaded["dataset"].upper() == "GTSRB":
        params_loaded["class_num"] = 43
        params_loaded["poison_train_batch_size"] = 32
        params_loaded["train_batch_size"] = 32
        params_loaded["poisoned_len"] = 4
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True, type=str)
    parser.add_argument("--no_of_adversaries", default=3, type=int)
    parser.add_argument("--poison_type", default="continue_poison", type=str)
    parser.add_argument("--attach", default="", type=str)
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--model_type", default="ResNet18", type=str)
    parser.add_argument("--dataset", default="CIFAR10", type=str)

    args = parser.parse_args()
    params_loaded = args_update(args)
    set_class_num_and_dataset_overrides(params_loaded)

    logger.info(f'params_loaded["resumed_model"] - {params_loaded["resumed_model"]}')

    if not params_loaded.get("run_id"):
        params_loaded["run_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info(f"run_id = {params_loaded['run_id']}")

    logger.info(f"Params: {params_loaded}")
    set_random_seed(params_loaded["seed"])

    dataloader = MSPDataloader(params_loaded)
    server = create_server(params_loaded, dataloader)
    benign_client = BenignClient(params_loaded, dataloader.train_dataloader, dataloader.test_dataloader)
    malicious_client = create_malicious_client(params_loaded, dataloader)

    if params_loaded.get("diagnostics", False):
        raise ValueError("Optional research diagnostics are not included in this release")

    for iteration in range(server.params["start_iteration"], server.params["end_iteration"]):
        round_started = time.perf_counter()
        logger.info(f"====================== Current Round: {iteration} ======================")
        server.pre_process(test_data=server.test_dataloader, iteration=iteration)

        weight_accumulator, weight_accumulator_by_client, aggregated_model_id = server.broadcast_upload(
            iteration=iteration,
            benign_client=benign_client,
            malicious_client=malicious_client,
        )

        server.aggregation(weight_accumulator=weight_accumulator, aggregated_model_id=aggregated_model_id)
        aggregation_mask = getattr(server, "last_aggregation_mask", aggregated_model_id)
        logger.info(f"aggregated_model:{aggregation_mask}")
        server.record_detection_metrics(iteration, aggregated_model_id)

        server.test_global_model(iteration=iteration, malicious_clients=malicious_client)
        server.save_model(iteration, malicious_client.trigger_set, malicious_client.mask_set)
        logger.info("|Round %s runtime %.2fs", iteration, time.perf_counter() - round_started)

    server.log_detection_summary()
    server.log_attack_summary()

