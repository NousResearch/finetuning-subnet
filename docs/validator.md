# Validator 

Validators download the models from 🤗 Hugging Face for each miner based on the Bittensor chain metadata and continuously evaluate them, setting weights based on the performance of each model against the data generated on the [Cortex.t subnet](https://taostats.io/subnets/netuid-18/).

You can view the entire validation system by reading the code in `neurons/validator.py`. Pseudocode for the validation system is as follows:
```python
    weights = zeros(256)
    while True:
        # Fetch random sample of batches to evaluate models on
        batches = get_random_sample_of_batches_from_coretex_subnet()
        
        # Fetch and or update models.
        models = get_and_update_models_from_miners()

        # Compute losses for each batch and each model
        model_losses = {}
        for model in models:
            for batch in batches:
                loss = get_loss_for_model_on_batch( model, batch )
                model_losses[ model ].append( loss )

        # Compute wins for models.
        model_wins = {}
        for model_a in models:
            for model_b in models:
                for i in len( batches )
                    # Determine if better model loss with relative block number boosting.
                    if iswin( model_losses[ model_a ][ i ], model_losses[ model_b ][ i ], block_a, block_b ):
                        model_wins[ model_a ] += 1
                            
        # End epoch.
        # Weights are computed based on the ratio of wins a model attains during the epoch.
        for model_i in models:
            weights[ model_i ] += model_wins[ model_i ] / sum( model_wins.values() )
        weights = softmax( weights / temperature, dim=0 )

        # Set weights on the chain.
        set_weights( weight )
```

The behaviour of `iswin( loss_a, loss_b, block_a, block_b)` function intentionally skews the win function to reward models which have been hosted earlier such that newer models are only better than others iff their loss is `epsilon` percent lower accoring to the following function. Currently `epsilon` is set to 1% and is a hyper parameter of the mechanism

```python
def iswin( loss_a, loss_b, block_a, block_b ):
    loss_a = (1 - constants.timestamp_epsilon) * loss_a if block_a < block_b else loss_a
    loss_b = (1 - constants.timestamp_epsilon) * loss_b if block_b < block_a else loss_b
    return loss_a < loss_b
```

It is important to note that this affects the game theoretics of the incentive landscape since miners should only update their model (thus updating their timestamp to a newer date) if they have achieved an `epsilon` better loss on average on the Falcon Refined Web dataset than their previous model. This undermines the obvious optimal strategy for miners to copy the publicly available models from other miners. They **can** and should copy other miners, but they will always obtain fewer wins compared to them until they also decrease their loss by `epsilon`.

# System Requirements

Validators will need enough disk space to store the model of every miner in the subnet. Each model (As of Jan 1st, 2024) is limited to 15 GB and 7B parameters, and the validator has cleanup logic to remove old models. It is recommended to have at least 500 GB of disk space.

Validators will need enough processing power to evaluate their model. As of Jan 1st, 2024 it is required to have a GPU with atleast 24 GB of VRAM.

# Getting Started

## Prerequisites

1. Clone the repo

```shell
git clone https://github.com/NousResearch/finetuning-subnet.git
```

2. Setup your python [virtual environment](https://docs.python.org/3/library/venv.html) or [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

3. Install the requirements. From your virtual environment, run
```shell
cd finetuning-subnet
python -m pip install -e .
```

4. Make sure you've [created a Wallet](https://docs.bittensor.com/getting-started/wallets) and [registered a hotkey](https://docs.bittensor.com/subnets/register-and-participate).

5. (Optional) Run a Subtensor instance:

Your node will run better if you are connecting to a local Bittensor chain entrypoint node rather than using Opentensor's. 
We recommend running a local node as follows and passing the ```--subtensor.network local``` flag to your running miners/validators. 
To install and run a local subtensor node follow the commands below with Docker and Docker-Compose previously installed.
```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor
docker compose up --detach
```
---

# Running the Validator

## With auto-updates

We highly recommend running the validator with auto-updates. This will help ensure your validator is always running the latest release, helping to maintain a high vtrust.

Prerequisites:
1. To run with auto-update, you will need to have [pm2](https://pm2.keymetrics.io/) installed.
2. Make sure your virtual environment is activated. This is important because the auto-updater will automatically update the package dependencies with pip.
3. Make sure you're using the main branch: `git checkout main`.

From the finetuning-subnet folder:
```shell
pm2 start --name finetune-vali-updater --interpreter python scripts/start_validator.py -- --pm2_name finetune-vali --wallet.name coldkey --wallet.hotkey hotkey [other vali flags]
```

This will start a process called `finetune-vali-updater`. This process periodically checks for a new git commit on the current branch. When one is found, it performs a `pip install` for the latest packages, and restarts the validator process (who's name is given by the `--pm2_name` flag)

## Without auto-updates

If you'd prefer to manage your own validator updates...

From the `finetuning-subnet` folder:
```shell
pm2 start python -- ./neurons/validator.py --wallet.name coldkey --wallet.hotkey hotkey
```

# Configuration

## Flags

The Validator offers some flags to customize properties, such as the device to evaluate on and the number of models to evaluate each step.
Of particular note is the `--attn_implementation` flag, which specifies the attention implementation. Those using newer CUDA-capable GPUs, installing [Flash Attention](https://github.com/Dao-AILab/flash-attention) with `pip install flash-attn` and passing `--attn_implementation flash_attention_2` will speed up model evaluation and reduce GPU memory usage.

You can view the full set of flags by running
```shell
python ./neurons/validator.py -h
```

## Test Running Validation

Test running validation:
```shell
python neurons/validator.py 
    --wallet.name YOUR_WALLET_NAME
    --wallet.hotkey YOUR_WALLET_HOTKEY 
    --device YOUR_CUDA DEVICE
    --wandb.off
    --offline
```
---