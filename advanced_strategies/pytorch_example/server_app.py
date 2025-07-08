"""pytorch-example: A Flower / PyTorch app."""

import os
from logging import WARN

import wandb
from datasets import load_dataset
from flwr.common import Context, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from pytorch_example.task import (
    Net,
    apply_eval_transforms,
    get_weights,
    set_weights,
    test,
)
from torch.utils.data import DataLoader

from .strategy import CustomFedAvg

USE_WANDB = False


def gen_evaluate_fn(
    testloader: DataLoader,
    device: str = "cpu",
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device=device)

        if USE_WANDB:
            wandb.log({"centralized_accuracy": accuracy}, step=server_round)
        return loss, {"centralized_accuracy": accuracy}

    return evaluate


def on_fit_config(server_round: int):
    """Construct `config` that clients receive when running `fit()`"""
    lr = 0.05
    # Enable a simple form of learning rate decay
    if server_round > 10:
        lr /= 2
    return {"lr": lr}


# Define metric aggregation function
def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]

    # Prep W&B
    global USE_WANDB
    USE_WANDB = context.run_config["use-wandb"]
    wandbtoken = context.run_config.get("wandb-token")
    if USE_WANDB:
        if not wandbtoken:
            log(
                WARN,
                "W&B token wasn't found. Set it by passing `--run-config=\"wandb-token='<YOUR-TOKEN>'\" to your `flwr run` command.",
            )
            USE_WANDB = False
        else:
            os.environ["WANDB_API_KEY"] = wandbtoken
            wandb.init(project="Flower-Simulation-with-strategies")

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Prepare dataset for central evaluation

    # This is the exact same dataset as the one downloaded by the clients via
    # FlowerDatasets. However, we don't use FlowerDatasets for the server since
    # partitioning is not needed.
    # We make use of the "test" split only
    global_test_set = load_dataset("zalando-datasets/fashion_mnist")["test"]

    testloader = DataLoader(
        global_test_set.with_transform(apply_eval_transforms),
        batch_size=32,
    )

    # Define strategy
    if not context.run_config["use-custom-strategy"]:
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_eval,
            initial_parameters=parameters,
            on_fit_config_fn=on_fit_config,
            evaluate_fn=gen_evaluate_fn(testloader),
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    else:
        # Achieve greater control by means of a custom strategy
        strategy = CustomFedAvg(
            run_config=context.run_config,
            use_wandb=USE_WANDB,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_eval,
            initial_parameters=parameters,
            on_fit_config_fn=on_fit_config,
            evaluate_fn=gen_evaluate_fn(testloader),
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
