"""pytorch-example-low-level: A low-level Flower / PyTorch app."""

import json
import os
import random
from collections import OrderedDict
from logging import INFO, WARN
from time import sleep, time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MessageType,
    RecordDict,
)
from flwr.common.logger import log
from flwr.server import Grid, ServerApp
from pytorch_example_low_level.task import (
    Net,
    apply_eval_transforms,
    create_run_dir,
    test,
)
from torch.utils.data import DataLoader

PROJECT_NAME = "FLOWER-advanced-pytorch-low-level"
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """A ServerApp that implements a for loop to define the stages in a round."""

    # Create run directory and save run-config
    save_path, run_dir = create_run_dir(context.run_config)

    # Initialize Weights & Biases if set
    use_wandb = context.run_config["use-wandb"]
    wandbtoken = context.run_config.get("wandb-token")
    if use_wandb:
        if not wandbtoken:
            log(
                WARN,
                "W&B token wasn't found. Set it by passing `--run-config=\"wandb-token='<YOUR-TOKEN>'\" to your `flwr run` command.",
            )
            use_wandb = False
        else:
            os.environ["WANDB_API_KEY"] = wandbtoken
            wandb.init(project="Flower-Simulation-with-messageapi")

    num_rounds = context.run_config["num-server-rounds"]
    batch_size = context.run_config["batch-size"]
    fraction_train = context.run_config["fraction-clients-train"]

    # Initialize global model
    global_model = Net()

    # Prepare global test set and dataloader
    global_test_set = load_dataset("zalando-datasets/fashion_mnist")["test"]

    testloader = DataLoader(
        global_test_set.with_transform(apply_eval_transforms),
        batch_size=batch_size,
    )

    # Keep track of best accuracy obtained (to save checkpoint when new best is found)
    best_acc_so_far = 0

    # A dictionary to store results as they come
    results = {"fed_accuracy": [], "fed_loss": []}

    for server_round in range(num_rounds):
        log(INFO, "")
        log(INFO, "🔄 Starting round %s/%s", server_round + 1, num_rounds)

        ### 1. Get IDs of nodes available
        node_ids = []
        while len(node_ids) == 0:
            # Wait until at least one node is available
            node_ids = list(grid.get_node_ids())
            sleep(1)

        # Sample uniformly
        num_sample = int(len(node_ids) * fraction_train)
        sampled_node_ids = random.sample(node_ids, num_sample)
        log(INFO, f"Sampled {len(sampled_node_ids)} out of {len(node_ids)} nodes.")

        ### 2. Create messages for Training
        messages = construct_train_or_eval_messages(
            global_model, grid, sampled_node_ids, MessageType.TRAIN, server_round
        )

        # Send messages and wait for all results
        replies = grid.send_and_receive(messages)
        log(INFO, "📥 Received %s/%s results (TRAIN)", len(replies), len(messages))

        ### 3. Aggregate received models
        updated_global_state_dict = aggregate_parameters_from_messages(replies)

        # Update global model
        global_model.load_state_dict(updated_global_state_dict)

        # Centrally evaluate global model and save checkpoint if new best is found
        best_acc_so_far = evaluate_global_model_centrally_and_save_results(
            global_model,
            testloader,
            save_path,
            server_round,
            best_acc_so_far,
            use_wandb,
        )

        ### 4. Query nodes for opt-in evaluation
        opt_in_node_ids = query_nodes_for_evaluation(node_ids, grid, server_round)

        ### 5. Prepare messages for evaluation
        messages = construct_train_or_eval_messages(
            global_model, grid, opt_in_node_ids, MessageType.EVALUATE, server_round
        )

        # Send messages and wait for all results
        replies = grid.send_and_receive(messages)
        log(INFO, "📥 Received %s/%s results (EVALUATE)", len(replies), len(messages))

        ### 6. Process results, save and log
        avg_accuracy, avg_loss = process_evaluation_responses(replies)
        results["fed_accuracy"].append(avg_accuracy)
        results["fed_loss"].append(avg_loss)
        if use_wandb:
            # Log federated metrics to W&B
            metrics = {
                "federated_accuracy": avg_accuracy,
                "federated_loss": avg_loss,
            }
            wandb.log(metrics, step=server_round)

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(results, fp)


def process_evaluation_responses(replies: List[Message]) -> Tuple[float]:
    """Extract metrics returned by `ClientApp`s when they ran their eval() method."""
    losses = []
    accuracies = []
    # Append all results
    for res in replies:
        if res.has_content():
            evaluate_results = res.content["clientapp-evaluate"]
            losses.append(evaluate_results["loss"])
            accuracies.append(evaluate_results["accuracy"])
    # Convert to NumPy arrays to easily extract mean/std
    losses = np.array(losses)
    accuracies = np.array(accuracies)
    log(
        INFO,
        f"📊 Federated evaluation -> loss: {losses.mean():.3f}±{losses.std():.3f} / "
        f"accuracy: {accuracies.mean():.3f}±{accuracies.std():.3f}",
    )
    return accuracies.mean(), losses.mean()


def evaluate_global_model_centrally_and_save_results(
    global_model,
    testloader,
    save_dir,
    serverapp_round,
    best_acc,
    use_wandb,
    device: str = "cpu",
) -> float:
    """Evaluate performance of global model on centralized tests set.

    Save a model checkpoint if a new best model is found. Saves loss/accuracy as JSON.
    """
    global_model.to(device)
    loss, accuracy = test(global_model, testloader, device=device)
    log(
        INFO,
        f"💡 Centrally evaluated model -> loss: {loss: .4f} /  accuracy: {accuracy: .4f}",
    )

    if use_wandb:
        # Log Centralized metrics to W&B
        metrics = {"centralized_accuracy": accuracy, "centralized_loss": loss}
        wandb.log(metrics, step=serverapp_round)

    if accuracy > best_acc:
        best_acc = accuracy
        # Save the PyTorch model
        file_name = save_dir/f"model_state_acc_{accuracy}_round_{serverapp_round}.pth"
        log(INFO, "🎉 New best global model found: %f -> %s", accuracy, file_name)
        torch.save(global_model.state_dict(), file_name)

    return best_acc


def aggregate_parameters_from_messages(messages: List[Message]) -> nn.Module:
    """Average all ParametersRecords sent by `ClientApp`s under the same key.

    Return a PyTorch model that will server as new global model.
    """

    state_dict_list = []
    # Get state_dicts from each message
    for msg in messages:
        if msg.has_error():
            continue
        # Extract ArrayRecord with the udpated model sent by the `ClientApp`
        # Note `updated_model_dict` is the key used by the `ClientApp`.
        state_dict_as_a_record = msg.content["updated_model_dict"]
        # Convert to PyTorch's state_dict and append
        state_dict_list.append(state_dict_as_a_record.to_torch_state_dict())

    # Initialize from first state_dict to accumulate sums
    new_global_dict = state_dict_list[0]

    # Iterate through each dictionary in the list
    for d in state_dict_list:
        for key, value in d.items():
            new_global_dict[key] = np.add(new_global_dict[key], value)

    # Now take the average
    for key in new_global_dict:
        new_global_dict[key] = new_global_dict[key] / len(state_dict_list)

    # Retun aggregated state_dict
    return OrderedDict(new_global_dict)


def query_nodes_for_evaluation(
    node_ids: List[int], grid: Grid, server_round
) -> List[int]:
    """Query nodes and filter those that respond positively.

    This function shows how to interfere with a `ClientApp`'s query method
    and use the respone message they send to construct a sub-set of node_ids
    that will be later used for another purpose. In this example the resulting
    list will contain the node IDs that will be sent the global model for its
    evaluation.
    """

    # Construct QUERY messages, the payload will carry just the current
    # timestamp for illustration purposes.
    payload = RecordDict()
    c_record = ConfigRecord({"timestamp": time()})
    payload["query-config"] = c_record

    messages = []
    # One message for each node
    for node_id in node_ids:
        message = Message(
            content=payload,
            message_type=MessageType.QUERY,  # will be processed by the `ClientApp`'s @app.query
            dst_node_id=node_id,
        )
        messages.append(message)

    # Send and wait for 5 seconds to receive answer
    # The `ClientApp` artificially adds a delay, so some messages won't arrive in time
    # and therefore those nodes will be left out.
    replies = grid.send_and_receive(messages, timeout=5)
    log(INFO, "📨 Received %s/%s results (QUERY)", len(replies), len(messages))

    # Construct list of node IDs based on responses that arrived in time with opt-in
    filter_node_ids = []
    for res in replies:
        if res.has_content():
            if res.content["query-response"]["opt-in"]:
                filter_node_ids.append(res.metadata.src_node_id)

    log(
        INFO,
        "✅ %s/%s nodes opted-in for evaluation (QUERY)",
        len(filter_node_ids),
        len(messages),
    )
    return filter_node_ids


def construct_train_or_eval_messages(
    global_model: nn.Module,
    grid: Grid,
    node_ids: List[int],
    msg_type: MessageType,
    server_round: int,
) -> Message:
    """Construct messages addressing a particular method of a `ClientApp`.

    This function receives a list of node IDs and a PyTorch model
    whose's state_dict will be sent to the `ClientApp`s. With `msg_type`
    you can specify whether this message will be processed by the `ClientApp`'s
    `train` or `evaluate` method.
    """

    # Constuct array record out of model's state_dict
    a_record = ArrayRecord(global_model.state_dict())

    # We can use a ConfigRecord to communicate config settings to the `ClientApp`
    # Implement a basic form of learning rate decay
    lr = 0.05 if server_round < 10 else 0.1 / 2
    c_record = ConfigRecord({"lr": lr})

    # The payload of the messages is an object of type RecordDict
    # It carries dictionaries of different types of records.
    # Note that you can add as many records as you wish
    # https://flower.ai/docs/framework/ref-api/flwr.common.RecordDict.html
    recordset = RecordDict(
        {"config": c_record, "global_model_record": a_record},
    )

    messages = []
    # One message for each node
    # Here we send the same message to all nodes, this is not a requirement
    for node_id in node_ids:
        message = Message(
            content=recordset,
            message_type=msg_type,
            dst_node_id=node_id,
        )
        messages.append(message)

    return messages
