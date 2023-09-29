from flwr.client import NumPyClient, start_numpy_client
from pytorch_lightning import Trainer, LightningModule, LightningDataModule, loggers
from collections import OrderedDict
import torch
import numpy as np
from typing import List, Dict
import argparse
from torch import nn
from pathlib import Path

from train import ResNetClassifier
from data_module import DrivingDataModule

import logging
from flwr.common import logger
logger.logger.setLevel(logging.INFO)


class FlowerClient(NumPyClient):
    def __init__(self, model: LightningModule, data_module: LightningDataModule, client_id: int=0):
        self.model = model
        self.train_loader, self.val_loader, self.test_loader = data_module.get_data_loaders()
        self.client_id = client_id
        self.logger = loggers.TensorBoardLogger(save_dir="lightning_logs/client{}".format(client_id))

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        print(f"[Client ID {self.client_id}] get_parameters")
        return _get_parameters(self.model)

    def fit(self, parameters, config: Dict):
        print(f"[Client ID {self.client_id}] fit")
        self.set_parameters(parameters)
        trainer = Trainer(max_epochs=20, accelerator="auto", enable_model_summary=False, log_every_n_steps=1, logger=self.logger)
        trainer.fit(self.model, self.train_loader, self.val_loader)
        metrics = trainer.callback_metrics
        return self.get_parameters(config), len(self.train_loader.dataset), {"accuracy": metrics['val_accuracy'].item()}

    def evaluate(self, parameters: List[np.ndarray], config: Dict):
        print(f"[Client {self.client_id}] evaluate")
        self.set_parameters(parameters)
        trainer = Trainer(enable_model_summary=False, logger=self.logger)
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"] if len(results) > 0 else np.nan
        accuracy = results[0]["test_accuracy"] if len(results) > 0 else np.nan
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        _set_parameters(self.model, parameters)


def _get_parameters(model: nn.Module):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

def _set_parameters(model: nn.Module, parameters) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def flower_client_fn(client_id: int, data_dir: str, seed: int, server_address: str, enable_ssh: bool) -> FlowerClient:
    data_module = DrivingDataModule(data_dir=data_dir, batch_size=32, client_id=client_id, seed=seed)
    data_module.setup()
    resnet_classifier_model = ResNetClassifier(weights=None)
    client = FlowerClient(resnet_classifier_model, data_module, client_id)
    certificates = Path(".cache/certificates/ca.crt").read_bytes() if enable_ssh else None
    start_numpy_client(
        server_address=server_address,
        client=client,
        root_certificates=certificates,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument("--client_id", default=0, type=int, help="client id")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--seed", default=1234, type=int, help="for random generator")
    parser.add_argument("--server", default="127.0.0.1:8080", type=str)
    parser.add_argument("--enable_ssh", type=bool, default=False)
    args = parser.parse_args()
    flower_client_fn(args.client_id, args.data, args.seed, args.server, args.enable_ssh)
