import copy
import random
import logging
import time

from tqdm import tqdm

from participants.servers.BasicServer import BasicServer
from utils.utils import model_dist_norm_var, update_weight_accumulator
from utils.collapse_diagnostics import CollapseDiagnostics

logger = logging.getLogger("logger")


class No_defense_Server(BasicServer):
    def __init__(self, params, dataloader):
        super(No_defense_Server, self).__init__(params, dataloader)

        # 查看No_defense_Server的所有参数

    def broadcast_upload(self, iteration, benign_client, malicious_client, **kwargs):

        logger.info(f"Training on global iteration {iteration} ")

        selected_clients_list, malicious_clients_list = self.select_clients(iteration)
        diagnostic = None
        if self.params.get("collapse_diagnostics", False):
            if getattr(self, "_collapse_diagnostics", None) is None:
                self._collapse_diagnostics = CollapseDiagnostics(self.params, self.test_dataloader)
            diagnostic = self._collapse_diagnostics
            self._collapse_iteration = iteration
            diagnostic.report(self.global_model, iteration, "global_before_local_training")
        ''' 记录当前的训练中，有多少个恶意客户端'''
        current_no_of_adversaries = 0
        for client_id in selected_clients_list:
            if client_id in malicious_clients_list:
                current_no_of_adversaries += 1


        weight_accumulator = self.create_weight_accumulator()  # 初始化权重累加器, dict类型
        weight_accumulator_by_client = []
        update_norm_list = []
        global_model_copy = self.create_global_model_copy()
        global_model = copy.deepcopy(self.global_model)
        aggregated_model_id = [1] * self.params["no_of_participants_per_iteration"]
        for client_id in tqdm(selected_clients_list):
            if client_id in malicious_clients_list:
                client = malicious_client
            else:
                client = benign_client
            client_train_data = self.train_dataloader[client_id]

            local_model = copy.deepcopy(self.global_model)

            for name, params in local_model.named_parameters():
                params.requires_grad = True

            local_model.train()
            updated_model = client.local_train(iteration, local_model, client_train_data, client_id, test_loader=self.test_dataloader)
            if diagnostic is not None:
                diagnostic.report(updated_model, iteration, "client_after_local_training", client_id,
                                  "malicious" if client_id in malicious_clients_list else "benign")
            update_norm = model_dist_norm_var(updated_model, global_model_copy)  # 计算更新距离全局模型的二范数

            update_norm_list.append(round(update_norm.item(), 6))

            weight_accumulator, single_wa = update_weight_accumulator(updated_model, copy.deepcopy(self.global_model),
                                                                      weight_accumulator)
            weight_accumulator_by_client.append(single_wa)
            del local_model

        for client_ind,client_id in enumerate(selected_clients_list):
            logger.info(f"Client {client_id} update norm: {update_norm_list[client_ind]}")
        return weight_accumulator, weight_accumulator_by_client, aggregated_model_id

    def aggregation(self, weight_accumulator, aggregated_model_id):
        result = super().aggregation(weight_accumulator, aggregated_model_id)
        diagnostic = getattr(self, "_collapse_diagnostics", None)
        if diagnostic is not None:
            diagnostic.report(self.global_model, self._collapse_iteration, "global_after_aggregation")
        return result
