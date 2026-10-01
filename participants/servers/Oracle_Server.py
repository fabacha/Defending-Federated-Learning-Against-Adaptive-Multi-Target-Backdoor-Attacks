import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from participants.servers.No_defense_Server import No_defense_Server


logger = logging.getLogger("logger")


class Oracle_Server(No_defense_Server):
    def __init__(self, params, dataloader):
        if params.get("oracle_final_only_access", False):
            if params.get("oracle_allow_mirage_local_train", False):
                raise ValueError("Final-only access conflicts with intermediate Mirage local training")
            if str(params.get("malicious_train_algo", "")).lower() != "mirage":
                raise ValueError("Final-only trigger evaluation currently supports Mirage only")
            passes = params.get("oracle_final_search_passes", 1)
            if isinstance(passes, bool) or int(passes) != passes or passes < 1:
                raise ValueError("oracle_final_search_passes must be a positive integer")
        super().__init__(params, dataloader)

    def broadcast_upload(self, iteration, benign_client, malicious_client, **kwargs):
        if self._should_run_mirage_locally(iteration):
            rng_state = self._capture_rng_state()
            try:
                self._run_shadow_mirage_training(iteration, malicious_client)
            finally:
                self._restore_rng_state(rng_state)

        return super(Oracle_Server, self).broadcast_upload(
            iteration,
            benign_client,
            malicious_client,
            **kwargs,
        )

    def _should_run_mirage_locally(self, iteration):
        if self.params.get("oracle_final_only_access", False):
            return False
        enabled = bool(self.params.get("oracle_allow_mirage_local_train", False))
        attack = str(self.params.get("malicious_train_algo", "")).lower()
        return enabled and attack == "mirage" and iteration in self.poisoned_iterations

    def test_global_model(self, iteration, malicious_clients):
        if not self.params.get("oracle_final_only_access", False):
            return super().test_global_model(iteration, malicious_clients)
        accuracy, loss = self.test_model_once(iteration, self.test_dataloader, is_poisoned=False)
        self.acc_list.append(accuracy)
        logger.info("[Oracle final-only] round=%s clean_accuracy=%.2f%% loss=%.4f", iteration, 100 * accuracy, loss)
        if iteration != int(self.params["end_iteration"]) - 1:
            logger.info("[Oracle final-only] No attacker access; adaptive ASR not evaluated this round")
            return
        if getattr(self, "_final_access_evaluated", False):
            return
        passes = int(self.params.get("oracle_final_search_passes", 1))
        rng_state = self._capture_rng_state()
        records = []
        try:
            for client_id in range(int(self.params["no_of_adversaries"])):
                stats = []
                for search_pass in range(passes):
                    snapshot = copy.deepcopy(self.global_model).eval()
                    snapshot.requires_grad_(True)
                    with torch.enable_grad():
                        trigger = malicious_clients.search_trigger(
                            snapshot, self.train_dataloader[client_id], client_id
                        )
                    if not torch.isfinite(trigger).all():
                        raise FloatingPointError("Nonfinite final-only Mirage trigger")
                    malicious_clients.trigger_set[client_id] = trigger.detach()
                    stats.append(copy.deepcopy(getattr(malicious_clients, "last_search_stats", {})))
                    logger.info("[Oracle final-only] attacker=%s pass=%s search_stats=%s", client_id, search_pass + 1, stats[-1])
                    del snapshot
                target = self.params["poison_label_swap"][client_id]
                asr, attack_loss = self.test_model_once(
                    iteration, self.test_dataloader, is_poisoned=True,
                    trigger=malicious_clients.trigger_set[client_id],
                    mask=malicious_clients.mask_set[client_id], label_swap=target,
                )
                self.acc_p_list[client_id].append(asr)
                records.append({"client": client_id, "target": target, "asr": float(asr),
                                "loss": float(attack_loss), "search_stats": stats})
                logger.info("[Oracle final-only] attacker=%s target=%s FINAL_ASR=%.2f%%", client_id, target, 100 * asr)
        finally:
            self._restore_rng_state(rng_state)
        report = {"protocol": "oracle_final_only", "iteration": iteration,
                  "clean_accuracy": float(accuracy), "search_passes": passes,
                  "outer_loops_per_pass": int(self.params["trigger_search_no_times"]),
                  "attackers": records,
                  "mean_asr": float(np.mean([record["asr"] for record in records])) if records else None,
                  "worst_asr": max((record["asr"] for record in records), default=None)}
        destination = Path(self.params["folder_path"]) / "oracle_final_only.json"
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("[Oracle final-only] final report: %s", json.dumps(report))
        self._final_access_evaluated = True

    def _run_shadow_mirage_training(self, iteration, malicious_client):
        adversary_count = int(self.params.get("no_of_adversaries", 0))
        for client_id in range(adversary_count):
            local_model = copy.deepcopy(self.global_model)
            for parameter in local_model.parameters():
                parameter.requires_grad = True
            local_model.train()
            malicious_client.local_train(
                iteration,
                local_model,
                self.train_dataloader[client_id],
                client_id,
                test_loader=self.test_dataloader,
            )
            del local_model

        logger.info(
            "Oracle ran Mirage local training for clients %s to refresh evaluation triggers; "
            "all resulting model updates were discarded",
            list(range(adversary_count)),
        )

    def _capture_rng_state(self):
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _restore_rng_state(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    def select_clients(self, iteration):
        first_benign_client = int(self.params["no_of_adversaries"])
        total_clients = int(self.params["no_of_total_participants"])
        participants_per_round = int(self.params["no_of_participants_per_iteration"])
        benign_client_ids = range(first_benign_client, total_clients)

        if participants_per_round > len(benign_client_ids):
            raise ValueError(
                "Oracle defense cannot select the requested number of benign clients: "
                f"requested={participants_per_round}, available={len(benign_client_ids)}"
            )

        selected_clients = random.sample(benign_client_ids, participants_per_round)
        self.last_selected_clients_list = list(selected_clients)
        self.last_malicious_clients_list = []
        logger.info(
            "Oracle defense excluded adversarial clients %s; selected benign clients: %s",
            list(range(first_benign_client)),
            selected_clients,
        )
        return selected_clients, []
