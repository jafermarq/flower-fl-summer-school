"""pytorch-example-low-level: A low-level Flower / PyTorch app."""

import random
import time

import torch
from flwr.client import ClientApp
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from pytorch_example_low_level.task import Net, load_data, test, train

# Flower ClientApp
app = ClientApp()


@app.train()
def train_fn(msg: Message, context: Context):
    """A method that trains the received model on the local train set."""

    # Initialize model
    model = Net()
    # Dynamically determine device (best for simulations)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load this `ClientApp`'s dataset
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # Extract model received from `ServerApp`
    state_dict = msg.content["global_model_record"].to_torch_state_dict()

    # apply to local PyTorch model
    model.load_state_dict(state_dict)

    # Get learning rate value sent from `ServerApp`
    lr = msg.content["config"]["lr"]
    # Train with local dataset
    _ = train(
        model,
        trainloader,
        context.run_config["local-epochs"],
        lr=lr,
        device=device,
    )

    # Send reply back to `ServerApp`
    reply = RecordDict({"updated_model_dict": ArrayRecord(model.state_dict())})
    # Return message
    return Message(content=reply, reply_to=msg)


@app.evaluate()
def eval_fn(msg: Message, context: Context):
    """A method that evaluates the received model on the local validation set."""

    # Initialize model
    model = Net()
    # Dynamically determine device (best for simulations)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load this `ClientApp`'s dataset
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, evalloader = load_data(partition_id, num_partitions)

    # Extract model received from `ServerApp`
    state_dict = msg.content["global_model_record"].to_torch_state_dict()

    # apply to local PyTorch model
    model.load_state_dict(state_dict)

    # Evaluate with local dataset
    loss, accuracy = test(
        model,
        evalloader,
        device=device,
    )

    # Put resulting metrics into a MetricsRecord
    # Send reply back to `ServerApp`
    reply = RecordDict(
        {"clientapp-evaluate": MetricRecord({"loss": loss, "accuracy": accuracy})}
    )
    # Return message
    return Message(content=reply, reply_to=msg)


@app.query()
def query(msg: Message, context: Context):
    """A basic query method that aims to exemplify some opt-in functionality.

    The node running this `ClientApp` reacts to an incomming message by returning
    a `True` or a `False`. If `True`, this node will be sampled by the `ServerApp`
    to receive the global model and do evaluation in its `@app.eval()` method.
    """

    # Inspect message
    c_record = msg.content["query-config"]
    # print(f"Received: {c_record = }")

    # Sleep for a random amount of time, will result in some nodes not
    # repling back to the `ServerApp` in time
    time.sleep(random.randint(0, 2))

    # Randomly set True or False as opt-in in the evaluation stage
    # Note the keys used for the records below are arbitrary, but both `ServerApp`
    # and `ClientApp` need to be aware of them.
    c_record_response = ConfigRecord(
        {"opt-in": random.random() > 0.5, "ts": time.time()}
    )
    reply_content = RecordDict({"query-response": c_record_response})

    # Return message
    return Message(content=reply_content, reply_to=msg)
