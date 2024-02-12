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
import typing
import torch
import bittensor as bt
import wandb
from torch.utils.data import IterableDataset
from wandb.apis.public.history import HistoryScan
from transformers import PreTrainedTokenizerBase
import constants
import time
import numpy as np
import math
from tqdm import tqdm

class CortexSubsetLoader(IterableDataset):
    def __init__(self, latest=True, random_seed: typing.Optional[int] = None,
                 max_samples=300, steps: typing.Optional[int]=1, progress=False,
                 retry_limit=10, page_size=100, running: typing.Optional[bool]=False,
                 cortex_project=constants.CORTEX_WANDB_PROJECT,
                 cortex_type=constants.CORTEX_WANDB_TYPE):
        api = wandb.Api(timeout=100)

        filters = [
            { "config.type": cortex_type }
        ]
        if running:
            filters.append( {"state": "running"} )
        runs = api.runs(cortex_project, filters={"$and": filters})

        retry_delay = 5  # Seconds to wait between retries
        attempt = 0

        generator = np.random.default_rng(seed=random_seed) if random_seed else None

        while attempt < retry_limit:
            try:
                run_order = list(range(len(runs)))

                if generator is not None:
                    generator.shuffle(run_order)

                self.buffer: typing.List[typing.Tuple[str, str]] = []
                self.selected_runs: typing.List[int] = []

                for run_index in tqdm(run_order, desc="Run", leave=False, disable=not progress):
                    run = runs[run_index]
                    self.selected_runs.append(run_index)

                    if latest:
                        last_step: int = run.lastHistoryStep
                    elif generator is not None:
                        last_step = int(generator.random() * run.lastHistoryStep)
                    else:
                        last_step = 0
                    max_step = last_step + 1
                    min_step = max(0, max_step - steps) if steps is not None else 0
                    history_scan = HistoryScan(run.client, run, min_step, max_step, page_size=page_size)
                    while True:
                        try:
                            sample = next(history_scan)
                            for uid in range(constants.CORTEX_MAX_UIDS):
                                try:
                                    prompt: typing.Optional[str] = sample[f"prompts.{uid}"]
                                    response: typing.Optional[str]  = sample[f"responses.{uid}"]
                                    if isinstance(prompt, str) and isinstance(response, str):
                                        if "as an ai language model" in prompt.lower():
                                            continue
                                        prompt = prompt.strip()
                                        response = response.strip()
                                        if len(prompt) > 0 and len(response) > 0:
                                            self.buffer.append((prompt, response))
                                            if len(self.buffer) == max_samples:
                                                return
                                except KeyError:
                                    pass
                        except StopIteration:
                            break
                bt.logging.warning(f"Did not collect {max_samples}, only got {len(self.buffer)}")
                return
            except:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch data, retrying. Attempt {attempt}/{retry_limit}"
                )
                if attempt < retry_limit:
                    time.sleep(retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

    def tokenize(self, tokenizer: PreTrainedTokenizerBase) -> typing.List[typing.Tuple[torch.Tensor, int]]:
        batches = []
        for prompt, response in self:
            conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            prompt_ids = tokenizer.apply_chat_template(
                [conversation[0]], truncation=True, max_length=constants.sequence_length,
                add_generation_prompt=True
            )
            ids = tokenizer.apply_chat_template(
                conversation, truncation=True, max_length=constants.sequence_length,
            )
            batches.append((torch.stack([torch.tensor(ids)]), len(prompt_ids)))
        return batches

    def __iter__(self):
        return self.buffer.__iter__()
