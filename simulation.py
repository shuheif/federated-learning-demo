import flwr as fl

from client import FlowerClient
from train import ResNetClassifier
from data_module import DrivingDataModule

DATA_DIR = '' # Insert path to your dataset here


def client_fn(client_id):
    data_module = DrivingDataModule(data_dir=DATA_DIR, batch_size=32, client_id=client_id, num_clients=2)
    data_module.setup()
    resnet_classifier_model = ResNetClassifier(pretrained=True)
    return FlowerClient(resnet_classifier_model, data_module)


hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=3),
)
assert (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) > 0.98
