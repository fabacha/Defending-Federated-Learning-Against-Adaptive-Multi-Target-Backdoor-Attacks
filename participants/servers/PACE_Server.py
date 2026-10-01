"""Independent same-round PACE filter with cumulative per-class access bans.

An eligible client's current upload is filtered immediately when its class
push dominates the runner-up. Each qualifying class event increments that
client's count for that class once. Counts survive clean participations,
nonparticipation and flags for other classes. A client reaching the threshold
for any class is removed from future training and aggregation. The
``pace_ban_model_access`` ablation controls whether it can still see each new
global model and refresh its trigger; the default denies access.
"""

import copy
import json
import logging
import math
import random
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from participants.servers.BasicServer import BasicServer
from utils.utils import model_dist_norm_var, update_weight_accumulator

logger = logging.getLogger("logger")


class PACE_Server(BasicServer):
    def __init__(self, params, dataloader):
        self.gap = float(params.get("pace_gap", 2.0))
        self.abs_floor = float(params.get("pace_abs_floor", 0.05))
        self.observe_only = params.get("pace_observe_only", False)
        self.ban_after = params.get("pace_ban_after", 3)
        self.replace_banned_slots = params.get("pace_ban_replace_slots", True)
        self.ban_model_access = params.get("pace_ban_model_access", False)
        if not math.isfinite(self.gap) or self.gap <= 1:
            raise ValueError("pace_gap must be finite and greater than one")
        if not math.isfinite(self.abs_floor) or self.abs_floor < 0:
            raise ValueError("pace_abs_floor must be finite and nonnegative")
        if not isinstance(self.observe_only, bool):
            raise ValueError("pace_observe_only must be a YAML boolean")
        if self.observe_only:
            raise ValueError("PACE requires pace_observe_only: false")
        if isinstance(self.ban_after, bool) or not isinstance(self.ban_after, int) or self.ban_after < 1:
            raise ValueError("pace_ban_after must be a positive integer")
        if not isinstance(self.replace_banned_slots, bool):
            raise ValueError("pace_ban_replace_slots must be a YAML boolean")
        if not isinstance(self.ban_model_access, bool):
            raise ValueError("pace_ban_model_access must be a YAML boolean")
        self.class_num = int(params["class_num"])
        self.participations = defaultdict(int)
        self.class_flag_counts = defaultdict(lambda: defaultdict(int))
        self.banned_clients = set()
        self.ban_round = {}
        self.ban_target = {}
        self.last_excluded_candidate_ids = []
        self.last_access_only_attacker_ids = []
        self.last_pace_records = []
        super().__init__(params, dataloader)
        logger.info("PACE gap=%g floor=%g ban_after=%d model_access_after_ban=%s",
                    self.gap, self.abs_floor, self.ban_after, self.ban_model_access)

    def _draw_candidates(self, iteration):
        """Use the normal selection schedule without logging banned candidates."""
        seed = self.params.get("client_sampling_seed")
        rng = random if seed is None else random.Random((int(seed) << 32) + int(iteration))
        poison_type = self.params.setdefault("poison_type", "continue_poison")
        total = int(self.params["no_of_total_participants"])
        per_round = int(self.params["no_of_participants_per_iteration"])
        adversaries = int(self.params["no_of_adversaries"])
        malicious = []
        if iteration in self.poisoned_iterations:
            if "continue_poison" in poison_type:
                malicious = list(range(adversaries))
                selected = malicious + rng.sample(range(adversaries, total), per_round - len(malicious))
            elif "full_random" in poison_type:
                selected = rng.sample(range(total), per_round)
                malicious = [cid for cid in selected if cid < adversaries]
            elif "sequential_poison" in poison_type:
                malicious = [iteration % adversaries]
                selected = malicious + rng.sample(range(adversaries, total),
                                                  per_round - len(malicious))
            else:
                raise ValueError(f"Unknown poison_type: {poison_type}")
        else:
            selected = rng.sample(range(total), per_round)
        return selected, malicious

    def select_clients(self, iteration):
        """Preserve the normal draw, then remove banned IDs before training.

        Keeping unbanned IDs from the base draw matches the comparison schedule
        until bans occur. Optional replacements come from the remaining eligible
        clients and never reintroduce a banned ID.
        """
        selected, malicious = self._draw_candidates(iteration)

        self.last_excluded_candidate_ids = [cid for cid in selected if cid in self.banned_clients]
        retained = [cid for cid in selected if cid not in self.banned_clients]
        removed = len(selected) - len(retained)
        replacements = []
        if removed and self.replace_banned_slots:
            seed = self.params.get("client_sampling_seed")
            rng = (random if seed is None else
                   random.Random(((int(seed) << 32) + int(iteration)) ^ 0x9E3779B97F4A7C15))
            poison_type = self.params.get("poison_type", "continue_poison")
            # Continuing/sequential attacks designate adversaries explicitly;
            # only full_random can draw an additional eligible attacker.
            benign_only = (iteration in self.poisoned_iterations and
                           "full_random" not in poison_type)
            lower = int(self.params["no_of_adversaries"]) if benign_only else 0
            pool = [cid for cid in range(lower, int(self.params["no_of_total_participants"]))
                    if cid not in self.banned_clients and cid not in selected]
            replacements = rng.sample(pool, min(removed, len(pool)))
        selected = retained + replacements
        if iteration in self.poisoned_iterations and "full_random" in self.params.get("poison_type", ""):
            malicious = [cid for cid in selected if cid < int(self.params["no_of_adversaries"])]
        else:
            malicious = [cid for cid in malicious if cid in selected]
        self.last_selected_clients_list = list(selected)
        self.last_malicious_clients_list = list(malicious)
        logger.info("[PACE] selected=%s malicious=%s excluded_candidates=%s replacements=%s",
                    selected, malicious, self.last_excluded_candidate_ids, replacements)
        return selected, malicious

    @torch.no_grad()
    def _head_push(self, updated_model, reference):
        heads = [(name, layer) for name, layer in updated_model.named_modules()
                 if isinstance(layer, torch.nn.Linear)]
        if not heads:
            raise ValueError("PACE requires a final Linear classifier")
        name, head = heads[-1]
        if head.out_features != self.class_num:
            raise ValueError("Final Linear output size must match class_num")
        prefix = name + "." if name else ""
        delta = head.weight.detach() - reference[prefix + "weight"]
        magnitude = delta.float().square().sum(dim=1)
        if head.bias is not None:
            magnitude += (head.bias.detach() - reference[prefix + "bias"]).float().square()
        result = magnitude.sqrt().cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError("PACE received a nonfinite classifier update")
        return result

    @torch.no_grad()
    def _round_events(self, values):
        events = torch.zeros_like(values, dtype=torch.bool)
        if len(values) < 2:
            return events
        leaders, indices = values.topk(2, dim=0)
        qualifies = (leaders[0] > self.abs_floor) & (leaders[0] >= self.gap * leaders[1].clamp_min(1e-9))
        events[indices[0], torch.arange(self.class_num)] = qualifies
        return events

    def _current_round_flags(self, round_r):
        present = list(round_r)
        if not present:
            return {}, {}
        values = torch.stack([round_r[client_id] for client_id in present])
        if values.shape != (len(present), self.class_num) or not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("PACE scores must be finite nonnegative class vectors")
        events = self._round_events(values)
        flags = {}
        event_by_client = {}
        for index, client_id in enumerate(present):
            classes = events[index].nonzero(as_tuple=True)[0].tolist()
            if classes:
                flags[client_id] = max(classes, key=lambda target: float(values[index, target]))
            event_by_client[client_id] = events[index].tolist()
        return flags, event_by_client

    def _record_events(self, client_id, event_classes, iteration):
        """Count each qualifying class at most once per participation.

        Other classes, clean participations, and absent rounds never erase a
        prior class count. The first class reaching the threshold triggers a
        prospective ban; the current upload is still handled by the filter.
        """
        counts = self.class_flag_counts[client_id]
        reached = []
        for target in sorted(set(event_classes)):
            target = int(target)
            if not 0 <= target < self.class_num:
                raise ValueError(f"Invalid flagged class: {target}")
            counts[target] += 1
            if counts[target] >= self.ban_after:
                reached.append(target)
        if reached and client_id not in self.banned_clients:
            target = reached[0]
            self.banned_clients.add(client_id)
            self.ban_round[client_id] = int(iteration)
            self.ban_target[client_id] = target
            logger.info("[PACE] Client %s permanently banned after %s cumulative flags for class %s in round %s",
                        client_id, counts[target], target, iteration)
            return True
        return False

    def _write_records(self, records):
        self.last_pace_records = records
        for record in records:
            logger.info("[PACE-AUDIT] %s", json.dumps(record, allow_nan=False))
        folder = self.params.get("folder_path")
        if folder:
            path = Path(folder) / "pace_audit.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")

    @staticmethod
    def _accumulate(accumulator, single):
        for name in accumulator:
            accumulator[name].add_(single[name])

    @contextmanager
    def _isolated_access_rng(self, iteration, client_id):
        """Keep the extra access-only search from changing client sampling/training RNG."""
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        seed = (int(self.params.get("seed", 123)) + 1000003 * int(iteration) + int(client_id)) % (2 ** 32)
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def _refresh_access_only_attackers(self, iteration, malicious_client, reference):
        """Broadcast to banned attackers without selecting or aggregating them."""
        self.last_access_only_attacker_ids = []
        if not self.ban_model_access:
            return
        adversaries = int(self.params["no_of_adversaries"])
        for client_id in sorted(self.banned_clients):
            if client_id >= adversaries:
                continue
            local = copy.deepcopy(reference)
            local.requires_grad_(True)
            refreshed_trigger = False
            with self._isolated_access_rng(iteration, client_id):
                if hasattr(malicious_client, "search_trigger"):
                    refreshed = malicious_client.search_trigger(
                        local, self.train_dataloader[client_id], client_id,
                        test_loader=self.test_dataloader,
                    )
                    malicious_client.trigger_set[client_id] = refreshed.detach()
                    refreshed_trigger = True
                elif hasattr(malicious_client, "_optimize_trigger"):
                    malicious_client._optimize_trigger(
                        local, self.train_dataloader[client_id], client_id,
                    )
                    refreshed_trigger = True
                # Fixed-pattern attacks have no trigger search to refresh.
            if refreshed_trigger:
                self.last_access_only_attacker_ids.append(client_id)
            del local
        logger.info("[PACE] access_only=%s refreshed_trigger_clients=%s; uploads_discarded",
                    sorted(self.banned_clients), self.last_access_only_attacker_ids)

    def broadcast_upload(self, iteration, benign_client, malicious_client, **kwargs):
        selected, malicious = self.select_clients(iteration)
        reference = self.global_model
        reference_state = reference.state_dict()
        round_r, updates_by_client, norms = {}, {}, {}

        # Both modes exclude banned clients from selection and aggregation. In
        # the access ablation they still receive the current model, so Mirage
        # can re-optimize its trigger before this round's ASR evaluation.
        self._refresh_access_only_attackers(iteration, malicious_client, reference)

        # select_clients has already removed every banned candidate.
        eligible = selected

        # Access control happens before any model copy or local_train call.
        for client_id in tqdm(eligible):
            client = malicious_client if client_id in malicious else benign_client
            local = copy.deepcopy(reference)
            local.requires_grad_(True)
            local.train()
            updated = client.local_train(iteration, local, self.train_dataloader[client_id],
                                         client_id, test_loader=self.test_dataloader)
            round_r[client_id] = self._head_push(updated, reference_state)
            norms[client_id] = float(model_dist_norm_var(updated, reference_state).item())
            _, single = update_weight_accumulator(updated, reference, self.create_weight_accumulator())
            updates_by_client[client_id] = single
            del local, updated

        flags, events = self._current_round_flags(round_r)
        candidates = set(flags)
        excluded = candidates
        newly_banned = {client_id for client_id in eligible
                        if self._record_events(
                            client_id,
                            [target for target, qualified in enumerate(events[client_id]) if qualified],
                            iteration,
                        )}
        accumulator = self.create_weight_accumulator()
        updates, weights, records = [], [], []
        target_labels = self.params.get("poison_label_swap", [])
        for client_id in selected:
            single = updates_by_client.get(client_id)
            if single is None:
                single = self.create_weight_accumulator()
            weight = int(client_id not in excluded)
            if weight:
                self._accumulate(accumulator, single)
            updates.append(single)
            weights.append(weight)
            self.participations[client_id] += 1
            others = [round_r[other] for other in eligible if other != client_id]
            rival = torch.stack(others).max(dim=0).values if others else torch.zeros(self.class_num)
            target = flags.get(client_id)
            target_counts = dict(sorted(self.class_flag_counts[client_id].items()))
            records.append({
                "round": int(iteration), "client": int(client_id), "observe_only": False,
                "decision_basis": "current_round",
                "decision_target": target,
                "sampling": self.params.get("poison_type", "continue_poison"),
                "head_push": round_r[client_id].tolist(),
                "largest_other_push": rival.tolist(),
                "dominance_ratio": (round_r[client_id] / rival.clamp_min(1e-9)).tolist(),
                "strike_event": events.get(client_id, [False] * self.class_num),
                "count_basis": "cumulative_per_class",
                "class_flag_counts": target_counts,
                "count_for_decision_target": target_counts.get(target, 0),
                "ban_after": self.ban_after,
                "ban_model_access": self.ban_model_access,
                "newly_banned": client_id in newly_banned,
                "banned": client_id in self.banned_clients,
                "ban_round": self.ban_round.get(client_id),
                "ban_trigger_class": self.ban_target.get(client_id),
                "model_access_denied": False,
                "participations": self.participations[client_id], "update_norm": norms.get(client_id),
                "would_exclude": client_id in excluded, "excluded": client_id in excluded,
                "flagged_target_before": target,
                "flagged_target_after": target,
                "role": "malicious" if client_id in malicious else "benign",
                "true_target": int(target_labels[client_id]) if client_id in malicious and client_id < len(target_labels) else None,
            })
        self._write_records(records)
        logger.info("[PACE] round=%s selected=%s trained=%s same_round_filtered=%s excluded_candidates=%s newly_banned=%s banned_total=%s access_only_attackers=%s",
                    iteration, len(selected), len(eligible), len(candidates),
                    self.last_excluded_candidate_ids, sorted(newly_banned), len(self.banned_clients),
                    self.last_access_only_attacker_ids)
        return accumulator, updates, weights

