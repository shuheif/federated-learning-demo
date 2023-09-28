from typing import Dict, List, Tuple
import flwr as fl
from flwr.common import ndarrays_to_parameters, logger, Metrics
import torch
import argparse
from pathlib import Path
import logging
logger.logger.setLevel(logging.DEBUG)

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

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    avg_accuracy = sum(accuracies) / sum(examples)
    print('avg accuracy', avg_accuracy)
    return {"avg_accuracy": avg_accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Host server")
    parser.add_argument("--ckpt", type=str, help="path to .ckpt file for initial weights")
    parser.add_argument("--address", type=str, help="address of server")
    args = parser.parse_args()
    model = ResNetClassifier.load_from_checkpoint(args.ckpt_path, map_location=DEVICE)
    model_parameters = ndarrays_to_parameters(val.cpu().numpy() for _, val in model.state_dict().items())
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1., # Sample 100% of available clients for training
        fraction_evaluate=1., # Sample 100% of available clients for evaluation
        on_fit_config_fn=fit_config,
        initial_parameters=model_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    hist = fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
