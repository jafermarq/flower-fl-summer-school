# Two Days of [Flower](https://flower.ai/)


👋 Welcome to the GitHub page for the 2-day Flower session at the [3rd Federated Machine Learning Summer School](https://school.flta-conference.org/index.php)!

This repository contains the materials for the hands-on sessions on both days. You'll also find links to resources (documentation, blogs, video tutorials, code examples) that complement the morning session slides.

## Agenda

| Day        | Time       | Sessions                                      |
|------------|------------|-----------------------------------------------|
| **Day 1**  | Morning    | - Welcome & Flower Keynote <br> - Flower Framework & Flower Datasets |
|            | Afternoon  | - (hands-on) Flower Simulation Runtime <br> - (hands-on) Flower Simulation on ResearchGrid |
| **Day 2**  | Morning    | - Recap from Day 1; Q&A <br> - DP and SecAgg with Flower <br> - Flower Deployment Runtime |
|            | Afternoon  | - (hands-on) MegaDataGrid challenge <br> - Wrap up and Q&A |

## Flower Resources

- [Flower GitHub Repository](https://github.com/adap/flower) ⭐️ Give us a star!
- Join [Flower Discuss](https://discuss.flower.ai/) to share what you're building or ask for support in your projects!
- Flower Documentation:  
  - [Simulation Runtime](https://flower.ai/docs/framework/how-to-run-simulations.html)  
  - [Deployment Runtime](https://flower.ai/docs/framework/deploy.html)  
- [Flower Datasets](https://flower.ai/docs/datasets/), including the [visualization tutorial](https://flower.ai/docs/datasets/tutorial-visualize-label-distribution.html)  
- [Flower Blogs](https://flower.ai/blog/)  
- [Flower Monthly](https://flower.ai/events/flower-monthly/) for updates on projects built with Flower

> [!TIP]
> Follow us on [X](https://twitter.com/flwrlabs), [LinkedIn](https://de.linkedin.com/company/flwrlabs) and [YouTube](https://www.youtube.com/@flowerlabs) to stay updated with the latest Flower news!

## Day 1: All-things Flower Simulations


The plan for the first hands-on session is to familiarize yourself with the basic components of a `FlowerApp` (i.e., `ServerApp` and `ClientApp`). We recommend exploring the [quickstart-pytorch (with strategies)](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) example or the [quickstart-pytorch (with Message API)](https://github.com/adap/flower/tree/main/examples/app-pytorch) example if you’d like to try the more flexible `Message API`.

Note that both examples accomplish roughly the same task: training a small CNN model for image classification using CIFAR-10.

Before moving on to the more *advanced* examples, ask yourself the following questions:

* Do I know how to use the `flwr run` command effectively? Do I know how to override the `RunConfig` via `--run-config` and stream logs with `--stream`? Do I know how to run a `FlowerApp` in Flower’s `ResearchGrid`?
* Do I understand how to use `Flower Datasets`, including changing the number of partitions and selecting a partitioner?
* Do I have a high-level understanding of how the `ServerApp <--> ClientApp` interaction works when using *strategies* versus directly using the `Message API`?
* Can I create a new `FlowerApp` using `flwr new`, and do I know how to run the simulation both locally and in Flower’s `ResearchGrid`?

If you feel confident about all the above points, feel free to dive into the advanced examples listed below. If you’re unsure about any of them, ask for help! As with the earlier examples, each advanced example is available in two versions: one using high-level *strategies* and another built directly on the low-level `Message API`. These are adapted from the existing [advanced-pytorch](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) example and slightly modified for this course.

> \[!NOTE]
> Choose the version of the advanced example you're most comfortable with—or go through both. Please note that the Day-2 challenge will use the `Message API` exclusively, so we strongly encourage you to understand how the [quickstart-pytorch (with Message API)](https://github.com/adap/flower/tree/main/examples/app-pytorch) works.

> \[!NOTE]
> The *advanced* examples below use [Weights & Biases](https://wandb.ai/site/) to log metrics such as accuracy and loss. If you don’t have an account yet, please create one (it’s free!). Once you have an account, make sure you have your token/key available so you can pass it via `--run-config` as shown below.

The *advanced* examples extend what you’ve seen in the `quickstart` examples by introducing a custom strategy, integrating popular logging tools like Weights & Biases, and demonstrating more sophisticated usage of the `Message API`. These examples use the [Fashion-MNIST](https://huggingface.co/datasets/zalando-datasets/fashion_mnist) dataset, partitioned using the [Dirichlet Partitioner](https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.DirichletPartitioner.html#flwr_datasets.partitioner.DirichletPartitioner) from `Flower Datasets`. By default, they use a relatively high `alpha=100`, meaning the resulting partitions are not strongly non-IID.

Each example includes a clear README and extensive inline comments. Don’t hesitate to reach out if anything is unclear!

```shell
# Choose the example
cd advanced_strategies # OR advanced_messageapi

# Install
pip install -e .

# Run locally (or run directly in the ResearchGrid)
flwr run .

# Run locally and logging to W&B
flwr run . --run-config="wandb-token='<YOUR-TOKEN>'"

# Login in ResearchGrid (note you should have registered in flower.ai first)
flwr login . researchgrid --stream

# Run in Flower's ResearchGrid
flwr run . researchgrid --stream --run-config="wandb-token='<YOUR-TOKEN>'"
```

### Assignment

- What is the accuracy out-of-the-box when simulating a federation of 10, 20, 50 or 100 clients?
- How does it affect tweaking the `alpha` parameter in the `DirichtletPartitioner` ? Does a small `alpha` result in a more IID or a more non-IID setting ?
- Try to design the best performing model when `alpha=0.1` !
- Try different partitioners from [Flower Datasets](https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html#module-flwr_datasets.partitioner)!
- You are done with the above ? Create your own Flower App via `flwr new`! 

## Day-2: MegaDataGrid Challenge

In the `MegaDataGrid Challenge`, you'll take on the role of an AI data scientist with access to a federation of 15 hospitals (Flower `SuperNodes`), each hosting image data from the `OrganAMNIST` dataset, part of the [MedMNIST](https://medmnist.com/) super-dataset. These 15 `SuperNodes` are interconnected via a Flower `SuperLink`.

**Your task?** Design a `FlowerApp` that trains the best model using the data available across all `SuperNodes`. You won’t have access to the raw data, nor will you know the quantity or class distribution within each `SuperNode`.

Your **starting point** is the `FlowerApp` located in the `megadata_challenge` directory. This app closely mirrors the [quickstart-pytorch (with Message API)](https://github.com/adap/flower/tree/main/examples/app-pytorch) example used on Day 1. The main differences are: (1) integration with `W&B` for logging, and (2) an adapted model suited for grayscale images and an 11-class dataset.

To win the challenge, you need to develop an improved combination of **model architecture**, **training loop**, and **data augmentation** techniques. Use Flower effectively to achieve the highest distributed validation accuracy in the fewest number of rounds.

> \[!NOTE]
> The `ClientApp` will run on `SuperNodes` contributed primarily by your Summer School peers. It is therefore recommended to use relatively small models that can train on CPU in under 20 seconds per local epoch. For reference, the `ClientApp` training loop in the starter code (in `megadata_challenge`) runs in less than 1 second.


### Connecting as a SuperNode

To form the federation of 15 `SuperNodes`, you can run one locally on your laptop. Follow these steps:

1. [Install Docker](https://docs.docker.com/desktop/) on your machine if you haven’t already. While `Docker` is not strictly required to run a Flower `SuperNode`, it simplifies the setup process significantly. You're encouraged to review the [Dockerfile](./Dockerfile) in the repository, which was used to build the Docker images you'll run.

2. Coordinate with the rest of the class to determine which `SuperNode` you will operate.

3. Launch the `SuperNode` by running the following command in your terminal:

    ```shell
    docker run --rm -it -p 9092:9092 jafermarq/supernode-dX # where X is 0...14
    ```

### Submitting your FlowerApp to the MegaDataGrid

```shell
cd megadata_challenge

# Install the app
pip install -e .

# Login in MeagDataGrid (note you should have registered in flower.ai first)
flwr login .

# Run the FlowerApp
flwr run . --stream --run-config="wandb-token='<YOUR-TOKEN>'"
```
