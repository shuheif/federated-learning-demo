from flwr.client import NumPyClient, start_numpy_client
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from collections import OrderedDict
import torch
import numpy as np
from typing import List, Dict
import argparse
from torch import nn

from train import ResNetClassifier
from data_module import DrivingDataModule


class FlowerClient(NumPyClient):
    def __init__(self, model: LightningModule, data_module: LightningDataModule, client_id: int=0):
        self.model = model
        self.train_loader, self.val_loader, self.test_loader = data_module.get_data_loaders()
        self.client_id = client_id

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        print(f"[Client ID {self.client_id}] get_parameters")
        return _get_parameters(self.model)

    def fit(self, parameters, config: Dict):
        print(f"[Client ID {self.client_id} fit, config: {config}")
        self.set_parameters(parameters)
        trainer = Trainer(max_epochs=1, accelerator="auto")
        trainer.fit(self.model, self.train_loader, self.val_loader)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict):
        print(f"[Client {self.client_id}] evaluate, config: {config}")
        self.set_parameters(parameters)
        results = Trainer().validate(self.model, self.test_loader)
        loss = results[0]["test_loss"]
        return loss, len(self.test_loader.dataset), {"loss": loss}

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        _set_parameters(self.model, parameters)


def _get_parameters(model: nn.Module):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

def _set_parameters(model: nn.Module, parameters) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def flower_client_fn(client_id: int, num_clients: int, data_dir: str) -> FlowerClient:
    data_module = DrivingDataModule(data_dir=data_dir, batch_size=32, client_id=client_id, num_clients=num_clients)
    data_module.setup()
    resnet_classifier_model = ResNetClassifier(pretrained=True)
    client = FlowerClient(resnet_classifier_model, data_module, client_id)
    start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument("--client_id", default=0, type=int, help="client id")
    parser.add_argument("--num_clients", default=2, type=int, help="number of clients")
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    flower_client_fn(args.client_id, args.num_clients, args.data_dir)
