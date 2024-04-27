# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import math
import os
import wandb
import torch
import random
import argparse
import constants
import typing
from model.model_updater import ModelUpdater
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
import finetune as ft
import bittensor as bt
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from finetune.mining import Actions
from utilities import utils
import datetime as dt

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# === Config ===
def get_config():
    """
    Set up and parse the command-line arguments to configure the system.

    The configuration is responsible for setting up the environment including
    the model path, device to use, and the bittensor wallet and logging configurations.

    Returns:
        A namespace object containing the configuration parameters.
    """

    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Does not launch a wandb run, does not send model to wandb, does not check if registered",
    )
    parser.add_argument(
        "--wandb_project", type=str, help="The wandb project to log to."
    )
    parser.add_argument("--wandb_entity", type=str, help="The wandb entity to log to.")
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="The hugging face repo id, which should include the org or user and repo name. E.g. jdoe/finetuned",
    )
    parser.add_argument(
        "--avg_loss_upload_threshold",
        type=float,
        default=0,  # Default to never uploading.
        help="The threshold for avg_loss the model must achieve to upload it to hugging face. A miner can only advertise one model, so it should be the best one.",
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join(constants.ROOT_DIR, "local-models/"),
        help="Where to download/save models for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device on which to run. cpu or cuda",
    )
    parser.add_argument(
        "--load_best",
        action="store_true",
        help="If set, the miner loads the best model from wandb to train off.",
    )
    parser.add_argument(
        "--load_uid",
        type=int,
        default=None,
        help="If passed loads the model under the specified uid.",
    )
    parser.add_argument(
        "--load_model_dir",
        type=str,
        default=None,
        help="If provided, loads a previously trained HF model from the specified directory",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=-1,
        help="Number of training epochs (-1 is infinite)",
    )
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate.")
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=32,
        help="The number of training accumulation steps.",
    )
    parser.add_argument(
        "--cortex_steps",
        type=int,
        default=5,
        help="Number of Cortex steps to sample data from",
    )
    parser.add_argument(
        "--cortex_samples_per_epoch",
        type=int,
        default=4096,
        help="Number of samples trained on per epoch",
    )
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        help="Implementation of attention to use",
    )
    parser.add_argument(
        "--netuid",
        type=str,
        default=constants.SUBNET_UID,
        help="The subnet UID.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="datatype to load model in, either bfloat16 or float16",
    )
    parser.add_argument(
        "--competition_id",
        type=str,
        default=constants.ORIGINAL_COMPETITION_ID,
        help="competition to mine for (use --list-competitions to get all competitions)"
    )
    parser.add_argument(
        "--list_competitions",
        action="store_true",
        help="Print out all competitions"
    )
    parser.add_argument(
        "--use_hotkey_in_hash",
        action="store_true",  # Defaults to False.
        help="If true, use the hotkey of the miner when generating the hash.",
    )


    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)

    return config


async def load_starting_model(
    actions: Actions, config: bt.config, metagraph: bt.metagraph,
    model_parameters: constants.CompetitionParameters
) -> typing.Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads the model to train based on the provided config."""

    # Initialize the model based on the best on the network.
    if config.load_best:
        # Get the best UID be incentive and load it.
        best_uid = ft.graph.best_uid(metagraph)
        model, tokenizer = await actions.load_remote_model(best_uid, metagraph, config.model_dir)
        bt.logging.success(
            f"Training with model from best uid: {best_uid}. Model={str(model)}"
        )
        return model, tokenizer

    # Initialize the model based on a passed uid.
    if config.load_uid is not None:
        # Sync the state from the passed uid.
        model, tokenizer = await actions.load_remote_model(
            config.load_uid, metagraph, config.model_dir
        )
        bt.logging.success(
            f"Training with model from uid: {config.load_uid}. Model={str(model)}"
        )
        return model, tokenizer

    # Check if we should load a model from a local directory.
    if config.load_model_dir:
        model, tokenizer = actions.load_local_model(config.load_model_dir, model_parameters)
        bt.logging.success(f"Training with model from disk. Model={str(model)}")
        return model, tokenizer

    raise RuntimeError("No starting model specified, pass either --load_best, --load_uid, or --load_model_dir")

async def main(config: bt.config):
    # Create bittensor objects.
    bt.logging(config=config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)

    # If running online, make sure the miner is registered, has a hugging face access token, and has provided a repo id.
    my_uid = None
    if not config.offline:
        my_uid = utils.assert_registered(wallet, metagraph)
        HuggingFaceModelStore.assert_access_token_exists()

    # Configure the stores and miner actions.
    miner_actions = ft.mining.Actions.create(config, wallet, subtensor)

    # Create a unique run id for this run.
    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = ft.mining.model_path(config.model_dir, run_id)
    os.makedirs(model_dir, exist_ok=True)

    use_wandb = False
    if not config.offline:
        if config.wandb_project is None or config.wandb_entity is None:
            bt.logging.warning(
                "Wandb project or entity not specified. This run will not be logged to wandb"
            )
        else:
            use_wandb = True

    block = metagraph.block.item()
    model_parameters = ModelUpdater.get_competition_parameters(config.competition_id)
    if not model_parameters:
        raise RuntimeError(
            f"No model parameters found for block {block}"
        )
    model_parameters.kwargs["torch_dtype"] = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    model_parameters.kwargs["attn_implementation"] = config.attn_implementation

    # Init model.
    model, tokenizer = await load_starting_model(miner_actions, config, metagraph, model_parameters)
    model = model.train()
    model = model.to(config.device)

    bt.logging.success(f"Saving model to path: {model_dir}.")
    miner_actions.save(model, tokenizer, model_dir)

    # Build optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    wandb_run = None

    # If using wandb, start a new run.
    if use_wandb:
        token = os.getenv("WANDB_API_KEY")
        if not token:
            raise ValueError(
                "To use Wandb, you must set WANDB_API_KEY in your .env file"
            )

        wandb.login(key=token)

        wandb_run = wandb.init(
            name=run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config={
                "uid": my_uid,
                "hotkey": wallet.hotkey.ss58_address,
                "run_name": run_id,
                "version": ft.__version__,
                "type": "miner",
            },
            allow_val_change=True,
        )
    else:
        bt.logging.warning(
            "Not posting run to wandb. Either --offline is specified or the wandb settings are missing."
        )

    # Start the training loop
    epoch_step = 0
    global_step = 0
    n_acc_steps = 0
    best_avg_loss = math.inf
    accumulation_steps = config.accumulation_steps

    try:
        while epoch_step < config.num_epochs or config.num_epochs == -1:
            # Initialize loss accumulator for the epoch
            epoch_loss = 0.0

            # Prepare the data loader with random pages for each epoch
            bt.logging.success(
                f"Loading {config.cortex_samples_per_epoch} pages for training this epoch"
            )
            loader = ft.dataset.CortexSubsetLoader(
                latest=False,
                random_seed=random.randint(0, 100000000),
                max_samples=config.cortex_samples_per_epoch,
                steps=config.cortex_steps,
                page_size=config.cortex_steps,
            )
            batches = loader.tokenize(tokenizer)

            # Enumerate over the data loader
            n_batches = 0
            optimizer.zero_grad()  # Initialize gradients to zero

            for i, (batch, _) in enumerate(batches):
                # Move the input batch to the device
                inputs = batch.to(model.device)

                # Forward pass: compute the model output and loss
                outputs = model(inputs, labels=inputs)

                loss = outputs.loss / accumulation_steps  # Scale loss
                loss.backward()  # Accumulate gradients

                if (i + 1) % accumulation_steps == 0:
                    n_acc_steps += 1
                    optimizer.step()  # Perform a single optimization step
                    optimizer.zero_grad()  # Clear gradients
                    bt.logging.success(
                        f"Step: {n_acc_steps} loss: {outputs.loss.detach().item()}"
                    )
                    if use_wandb:
                        wandb_run.log(
                            {"loss": outputs.loss.detach(), "n_batches": n_batches},
                            step=n_acc_steps,
                        )

                torch.cuda.empty_cache()

                n_batches += 1
                global_step += 1
                epoch_loss += outputs.loss.detach().item()

            # Calculate the average loss for the epoch
            avg_loss = epoch_loss / n_batches

            # Log the average loss for the epoch
            bt.logging.success(f"Epoch: {epoch_step} average loss: {avg_loss}")
            epoch_step += 1

            # Check if the average loss of this epoch is the best we've seen so far
            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss  # Update the best average loss

                bt.logging.success(f"New best average loss: {best_avg_loss}.")

                # Save the model to your mining dir.
                bt.logging.success(f"Saving model to path: {model_dir}.")
                miner_actions.save(model, tokenizer, model_dir)

        bt.logging.success("Finished training")
        # Push the model to your run.
        if not config.offline:
            if best_avg_loss < config.avg_loss_upload_threshold:
                bt.logging.success(
                    f"Trained model had a best_avg_loss of {best_avg_loss} which is below the threshold of {config.avg_loss_upload_threshold}. Uploading to hugging face. "
                )

                # First, reload the best model from the training run.
                model_to_upload, tokenizer_to_upload = miner_actions.load_local_model(model_dir, model_parameters)
                await miner_actions.push(model_to_upload, tokenizer_to_upload, model_parameters, use_hotkey_in_hash=config.use_hotkey_in_hash)
            else:
                bt.logging.success(
                    f"This training run achieved a best_avg_loss={best_avg_loss}, which did not meet the upload threshold. Not uploading to hugging face."
                )
        else:
            bt.logging.success(
                "Not uploading to hugging face because --offline was specified."
            )

    finally:
        # Important step.
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    # Parse and print configuration
    config = get_config()
    if config.list_competitions:
        print(constants.COMPETITION_SCHEDULE)
    else:
        print(config)
        asyncio.run(main(config))
