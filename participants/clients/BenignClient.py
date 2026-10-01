"""Standard benign local training used by the reported server defenses."""

import torch

from participants.clients.BasicClient import BasicClient


class BenignClient(BasicClient):
    def __init__(self, params, train_dataloader, test_dataloader):
        super().__init__(params, train_dataloader, test_dataloader)
        local_method = str(params.get("local_defense_method", "none")).lower()
        if local_method not in {"none", ""}:
            raise ValueError(f"Unsupported local_defense_method in this release: {local_method}")

    def local_train(self, iteration, model, train_loader, client_id, **kwargs):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.get_lr(iteration),
            momentum=self.params["poisoned_momentum"],
            weight_decay=self.params["poisoned_weight_decay"],
        )
        for _ in range(self.params["benign_retrain_no_times"]):
            for inputs, labels in train_loader:
                inputs = inputs.to(self.params["run_device"])
                labels = labels.to(self.params["run_device"])
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        return model
