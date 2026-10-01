import copy
import datetime
import logging
import os

import math

import numpy as np
import torch
import yaml
from colorama import Fore

logger = logging.getLogger("logger")

# 创建一个全局变量static_args，用于保存命令行参数
global static_args
static_args = None


def configure_flcert_cuda_determinism(params):
    """Set cuBLAS workspace before the first CUDA operation in deterministic FLCert-D."""
    defense = str(params.get("defense_method", "")).lower().replace("-", "_")
    if defense not in ("flcert", "flcert_d") or not params.get("flcert_deterministic", True):
        return
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif current not in (":4096:8", ":16:8"):
        raise ValueError(
            "Deterministic FLCert-D requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "or :16:8; set it before starting Python."
        )


def args_update(args=None, mkdir=True):
    with open(f"./{args.params}", "r", encoding="utf-8") as f:
        new_args = yaml.safe_load(f)
    new_args.update(vars(args))
    configure_flcert_cuda_determinism(new_args)
    run_device = torch.device("cpu")
    if torch.cuda.is_available():
        run_device = torch.device(f"cuda:{new_args['gpu_id']}")
        print(torch.cuda.get_device_name(run_device))
    elif torch.backends.mps.is_available():
        run_device = torch.device("mps")

    new_args["run_device"] = run_device
    # 保留位数的默认处理
    new_args["round_ndigits"] = 8 if "round_ndigits" not in new_args.keys() else new_args[
        "round_ndigits"]

    print(Fore.GREEN + f"Running on device: {run_device}" + Fore.RESET)

    current_time = datetime.datetime.now().strftime("%b.%d_%H.%M.%S")
    new_args["folder_path"] = f"./saved_logs/{new_args['attach']}_{current_time}"
    new_args["save_on_iteration"].append(new_args["end_iteration"] - 1)
    if mkdir:
        try:
            os.makedirs(new_args["folder_path"])
        except FileExistsError:
            logger.info("Folder already exists")
        logger.addHandler(logging.FileHandler(filename=f"{new_args['folder_path']}/log.txt"))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logger.info(f"current path:{new_args['folder_path']}")
    # 给全局变量赋值
    global static_args
    static_args = new_args
    return new_args


def poisoned_batch_injection(batch, trigger, mask, is_eval=False, client_id=0, label_swap=None, mode = "all"):
    '''
    对batch数据进行投毒，并返回新的batch数据

    :param batch: 需要投毒的batch
    :param trigger: trigger tensor, shape为(channel, height, width)
    :param mask: mask tensor, shape为(channel, height, width)
    :param is_eval: 是否为eval模式
    :param client_id: 客户端id (for attacker)
    :param label_swap: target label (for attacker)
    :param mode: 模式，是否在注入时排除干净目标类, value: "all"/"escape_clean"
    :return: poisoned batch
    '''
    if label_swap is None:
        label_swap = static_args["poison_label_swap"][client_id]
    data, label = copy.deepcopy(batch)

    if mode == "all" and is_eval == False:
        # The final client batch can be shorter than poisoned_len when drop_last=False.
        poison_indices = list(range(min(static_args["poisoned_len"], len(label))))
    elif mode == "escape_clean" and is_eval == False:
        poison_indices = np.nonzero(label != label_swap)[:static_args["poisoned_len"]].ravel()
    elif is_eval == True:
        # 生成一个长度为len(label)的列表

        poison_indices = list(range(len(label)))
    else:
        raise ValueError("mode should be 'all' or 'escape_clean'")


    # 如果model == "escap_clean"， 那么poison_indices 为clean_indices中前poisoned_len个不为true的索引

    poisoned_len = static_args["poisoned_len"] if not is_eval else len(label)
    data = data.to(static_args["run_device"])
    trigger = trigger.to(static_args["run_device"])
    mask = mask.to(static_args["run_device"])
    if static_args["poisoned_pattern_choose"] == 1:  # 1 -> pixel block
        data[poison_indices] = trigger * mask + (1 - mask) * data[poison_indices]
    elif static_args["poisoned_pattern_choose"] == 2:  # 1 -> blend trigger
        data[poison_indices] = trigger * static_args["blend_alpha"] + (1 - static_args["blend_alpha"]) * data[
                                                                                                        poison_indices]
    label[poison_indices] = label_swap
    return data, label


def model_dist_norm_var(model, target_params_variables, norm=2):
    '''
    计算model 和 target_params_variables之间的距离，默认使用2范数
    :param self:
    :param model:
    :param target_params_variables:
    :param norm:
    :return: model和target_params_variables之间的距离
    '''
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.zeros(size, device=static_args["run_device"], dtype=torch.float)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
                layer - target_params_variables[name]).view(-1)
        size += layer.view(-1).shape[0]

    return torch.norm(sum_var, norm)


def model_dist_norm(model, target_params):
    squared_sum = 0
    for name, layer in model.named_parameters():
        squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
    return math.sqrt(squared_sum)


def update_weight_accumulator(model, global_model, weight_accumulator, weight = 1.0):
    '''
    计算模型更新的梯度，并累加到weight_accumulator中


    :param model:
    :param global_model:
    :param weight_accumulator:
    :return: weight_accumulator (当前权重累加), single_weight_accumulator (当前模型更新)
    '''
    single_weight_accumulator = dict()
    for name, data in model.state_dict().items():
        single_weight_accumulator[name] = data - global_model.state_dict()[name]
        try:
            weight_accumulator[name].add_((data - global_model.state_dict()[name]) * weight)
        except RuntimeError as e:
            if single_weight_accumulator[name].dtype == torch.int64:
                weight_accumulator[name].add_((data - global_model.state_dict()[name]))
    return weight_accumulator, single_weight_accumulator


