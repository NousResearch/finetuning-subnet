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

# Tools for performing validation over models.

import math
import torch
import typing
import constants
import traceback
import bittensor as bt


def iswin(loss_i, loss_j, block_i, block_j, current_block):
    """
    Determines the winner between two models based on the epsilon adjusted loss.

    Parameters:
        loss_i (float): Loss of uid i on batch
        loss_j (float): Loss of uid j on batch.
        block_i (int): Block of uid i.
        block_j (int): Block of uid j.
    Returns:
        bool: True if loss i is better, False otherwise.
    """
    # Adjust loss based on timestamp and pretrain epsilon
    block_delta_i = current_block - block_i
    if block_delta_i <= constants.timestamp_decay_start:
        epsilon_i = constants.timestamp_epsilon
    elif block_delta_i > constants.timestamp_decay_start and (block_delta_i - constants.timestamp_decay_start) <= constants.timestamp_decay_period:
        epsilon_factor_i = 1 - ((block_delta_i - constants.timestamp_decay_start) / constants.timestamp_decay_period)
        epsilon_i = (epsilon_factor_i * (constants.timestamp_epsilon - constants.timestamp_epsilon_min)) + constants.timestamp_epsilon_min
    else:
        epsilon_i = constants.timestamp_epsilon_min

    block_delta_j = current_block - block_j
    if block_delta_j <= constants.timestamp_decay_start:
        epsilon_j = constants.timestamp_epsilon
    elif block_delta_j > constants.timestamp_decay_start and (block_delta_j - constants.timestamp_decay_start) <= constants.timestamp_decay_period:
        epsilon_factor_j = 1 - ((block_delta_j - constants.timestamp_decay_start) / constants.timestamp_decay_period)
        epsilon_j = (epsilon_factor_j * (constants.timestamp_epsilon - constants.timestamp_epsilon_min)) + constants.timestamp_epsilon_min
    else:
        epsilon_j = constants.timestamp_epsilon_min

    if block_i < block_j:
        loss_i = (1 - epsilon_i) * loss_i

    if block_j < block_i:
        loss_j = (1 - epsilon_j) * loss_j

    return loss_i < loss_j


def compute_wins(
    uids: typing.List[int],
    losses_per_uid: typing.Dict[int, typing.List[float]],
    uid_to_block: typing.Dict[int, int],
    current_block: int
):
    """
    Computes the wins and win rate for each model based on loss comparison.

    Parameters:
        uids (list): A list of uids to compare.
        losses_per_uid (dict): A dictionary of losses for each uid by batch.
        batches (List): A list of data batches.
        uid_to_block (dict): A dictionary of blocks for each uid.

    Returns:
        tuple: A tuple containing two dictionaries, one for wins and one for win rates.
    """
    wins = {uid: 0 for uid in uids}
    win_rate = {uid: 0 for uid in uids}
    for i, uid_i in enumerate(uids):
        total_matches = 0
        block_i = uid_to_block[uid_i]
        for j, uid_j in enumerate(uids):
            if i == j:
                continue
            block_j = uid_to_block[uid_j]
            batches_i = len(losses_per_uid[uid_i])
            batches_j = len(losses_per_uid[uid_j])
            for batch_idx in range(0, min(batches_i, batches_j)):
                loss_i = losses_per_uid[uid_i][batch_idx]
                loss_j = losses_per_uid[uid_j][batch_idx]
                wins[uid_i] += 1 if iswin(loss_i, loss_j, block_i, block_j, current_block) else 0
                total_matches += 1
        # Calculate win rate for uid i
        win_rate[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0

    return wins, win_rate


def compute_losses(
    model, batches: typing.List[typing.Tuple[torch.Tensor, int]], device: str
) -> typing.List[float]:
    """
    Computes the losses for a given model on provided batches.

    Parameters:
        model (torch.nn.Module): The model for which losses are to be computed.
        batches (dict): A list of batches and the associated lengths of the "prompt" section.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').

    Returns:
        dict: A dictionary with page indices as keys and lists of loss values as values.
    """
    # Iterate over each page and corresponding batches
    losses = []
    with torch.inference_mode():
        model.to(device)
        model.eval()
        for inputs, prompt_len in batches:
            try:
                inputs = inputs.to(device)
                labels = inputs.clone()
                labels[:, :prompt_len] = -100 # Only calculate loss on response
                outputs = model(inputs, labels=labels)
                loss = outputs.loss.item()  # Extract scalar loss value
                losses.append(loss)
            except Exception as e:
                bt.logging.error(f"Exception occurred: {e}")
                traceback.print_exc()  # Print the stack trace
                losses.append(math.inf)  # Use infinity to indicate failure

    return losses
