import flwr as fl


def main() -> None:
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1., # Sample 100% of available clients for training
        fraction_evaluate=1., # Sample 100% of available clients for evaluation
    )
    # Start Flower server for three rounds of federated learning
    hist = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
    assert (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) > 0.98


if __name__ == "__main__":
    main()
