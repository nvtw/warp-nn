# Variational Autoencoder (VAE) Example

The example demonstrates how to train and test a Variational AutoEncoder (VAE) on the MNIST dataset.
It is adapted from the [PyTorch’s VAE example](https://github.com/pytorch/examples/tree/main/vae),
that is an improved implementation of the [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) paper.

## Run the example

### Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

```txt
usage: main.py [-h] [--batch-size BATCH_SIZE] [--device DEVICE] [--epochs EPOCHS] [--seed SEED]

Variational Autoencoder (VAE) - MNIST Example

options:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Batch size for training/testing (default: 128)
  --device DEVICE       Device to use (default: cuda)
  --epochs EPOCHS       Number of epochs to train (default: 10)
  --seed SEED           Random seed (default: 42)
```

### Run

To run the example, execute:

```bash
python main.py
```

After execution, images and metrics will be saved in the `results` directory.
