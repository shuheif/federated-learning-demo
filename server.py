from typing import Dict
import flwr as fl
from flwr.common import ndarrays_to_parameters
import torch
import argparse
from pathlib import Path

from train import ResNetClassifier

NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fit_config(server_round: int) -> Dict[str, str]:
    config = {
        "batch_size": 16,
        "current_round": server_round,
        "local_epochs": 2,
    }
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower server")
    parser.add_argument("--ckpt_path", type=str, help="path to .ckpt file for initial weights")
    args = parser.parse_args()
    model = ResNetClassifier.load_from_checkpoint(args.ckpt_path, map_location=DEVICE)
    model_parameters = ndarrays_to_parameters(val.cpu().numpy() for _, val in model.state_dict().items())
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1., # Sample 100% of available clients for training
        fraction_evaluate=1., # Sample 100% of available clients for evaluation
        on_fit_config_fn=fit_config,
        initial_parameters=model_parameters,
    )
    hist = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        certificates=(
            Path(".cache/certificates/ca.crt").read_bytes(),
            Path(".cache/certificates/server.pem").read_bytes(),
            Path(".cache/certificates/server.key").read_bytes(),
        ),
    )
    assert (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) > 0.98
