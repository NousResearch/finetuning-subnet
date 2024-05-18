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
from substrateinterface import Keypair
import wandb
from torch.utils.data import IterableDataset
from wandb.apis.public.history import HistoryScan
from transformers import PreTrainedTokenizerBase
import constants
import time
import numpy as np
from tqdm import tqdm

class CortexSubsetLoader(IterableDataset):
    """
    A dataset loader for fetching subsets of data from WandB Cortex project runs.
    
    Args:
        subtensor (bt.subtensor, optional): Bittensor subtensor instance.
        latest (bool): Whether to fetch the latest data.
        random_seed (int, optional): Random seed for shuffling.
        max_samples (int): Maximum number of samples to fetch.
        steps (int, optional): Number of steps to fetch.
        progress (bool): Whether to display progress bars.
        retry_limit (int): Number of retry attempts for fetching data.
        page_size (int): Page size for fetching data from WandB.
        running (bool, optional): Whether to fetch only running runs.
        cortex_project (str): Name of the WandB Cortex project.
        cortex_type (str): Type of the Cortex project.
    """
    
    def __init__(
        self,
        subtensor: typing.Optional[bt.subtensor] = None,
        latest=True,
        random_seed: typing.Optional[int] = None,
        max_samples=300,
        steps: typing.Optional[int] = 1,
        progress=False,
        retry_limit=10,
        page_size=100,
        running: typing.Optional[bool] = False,
        cortex_project=constants.CORTEX_WANDB_PROJECT,
        cortex_type=constants.CORTEX_WANDB_TYPE,
    ):
        self.api = wandb.Api(timeout=100)
        self.filters = [{"config.type": cortex_type}]
        self.subtensor = subtensor

        if running:
            self.filters.append({"state": "running"})
        self.runs = self.api.runs(cortex_project, filters={"$and": self.filters})
        self.retry_delay = 5  # Seconds to wait between retries
        self.max_samples = max_samples
        self.steps = steps
        self.progress = progress
        self.retry_limit = retry_limit
        self.page_size = page_size
        self.latest = latest
        self.generator = np.random.default_rng(seed=random_seed) if random_seed is not None else None

        self.run_order = list(range(len(self.runs)))
        if self.generator is not None:
            self.generator.shuffle(self.run_order)

        self.last_steps = []
        for run in self.runs:
            if self.latest:
                last_step: int = run.lastHistoryStep
            else:
                last_step = int(self.generator.random() * run.lastHistoryStep) if self.generator is not None else 0
            self.last_steps.append(last_step)

        self.buffer: typing.List[typing.Tuple[str, str]] = []
        self.selected_runs: typing.List[int] = []

        self.fetch_data()

    def fetch_data(self):
        """
        Fetches data from the WandB Cortex project runs with retry logic.
        """
        attempt = 0
        while attempt < self.retry_limit:
            try:
                self.process_runs()
                if len(self.buffer) < self.max_samples:
                    bt.logging.warning(f"Did not collect {self.max_samples}, only got {len(self.buffer)}")
                return
            except Exception as e:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch data: {e}. Retrying in {self.retry_delay} seconds. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(
                        f"Maximum retry limit reached. Unable to fetch data. Error: {e}"
                    )
                    raise

    def process_runs(self):
        """
        Processes each run to verify that it belongs to a validator w/ minimum stake and then collect samples from the run.
        """
        for run_index in tqdm(self.run_order, desc="Run", leave=False, disable=not self.progress):
            run = self.runs[run_index]
            if run.config:
                self.verify_and_select_run(run, run_index)
                self.collect_samples(run, run_index)
                if len(self.buffer) >= self.max_samples:
                    return

    def verify_and_select_run(self, run, run_index):
        """
        Verifies the run by checking that it is signed by an active Bittensor validator that meets minimum stake criteria.
        
        Args:
            run: The WandB run to verify.
            run_index: The index of the run in the list of runs.
        """
        try:
            id = run.id
            hotkey = run.config.get("hotkey")
            signature = run.config.get("signature")
            if id and hotkey and signature:
                keypair = Keypair(ss58_address=hotkey)
                verified = keypair.verify(id.encode(), bytes.fromhex(signature))
                if verified and self.subtensor is not None:
                    stake = self.subtensor.get_total_stake_for_hotkey(hotkey)
                    stake_int = int(stake)
                    if stake_int > 25000000000000:
                        self.selected_runs.append(run_index)
        except Exception as e:
            bt.logging.warning(f"Error verifying run {run.id}: {e}")

    def collect_samples(self, run, run_index):
        """
        Collects samples from the verified runs.
        
        Args:
            run: The WandB run to collect samples from.
            run_index: The index of the run in the list of runs.
        """
        last_step = self.last_steps[run_index]
        max_step = last_step + 1
        min_step = max(0, max_step - self.steps) if self.steps is not None else 0
        history_scan = HistoryScan(run.client, run, min_step, max_step, page_size=self.page_size)
        while True:
            try:
                sample = next(history_scan)
                if sample and sample.get("modality") == "text":
                    self.extract_and_store_prompts(sample)
                    if len(self.buffer) >= self.max_samples:
                        return
            except StopIteration:
                break
            except Exception as e:
                bt.logging.warning(f"Error collecting samples from run {run.id}: {e}")
                break

    def extract_and_store_prompts(self, sample):
        """
        Extracts and stores prompts and responses from the sample.
        
        Args:
            sample: The sample to extract prompts and responses from.
        """
        for uid in range(constants.CORTEX_MAX_UIDS):
            prompt: typing.Optional[str] = sample.get(f"prompts.{uid}")
            response: typing.Optional[str] = sample.get(f"responses.{uid}")
            score: typing.Optional[float] = sample.get(f"scores.{uid}")
            if prompt and response:
                prompt = prompt.strip()
                response = response.strip()
                if len(prompt) > 0 and len(response) > 0 and not any(x in response for x in constants.UNWANTED_PHRASES):
                    if score and isinstance(score, float) and score > 0.0:
                        self.buffer.append((prompt, response))
                    elif not score:
                        self.buffer.append((prompt, response))

    def tokenize(self, tokenizer: PreTrainedTokenizerBase) -> typing.List[typing.Tuple[torch.Tensor, int]]:
        """
        Tokenizes the collected prompts and responses.
        
        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenizing the data.
        
        Returns:
            List[Tuple[torch.Tensor, int]]: A list of tokenized prompts and responses.
        """
        batches = []
        for prompt, response in self:
            conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            try:
                prompt_ids = tokenizer.apply_chat_template(
                    [conversation[0]], truncation=True, max_length=constants.sequence_length,
                    add_generation_prompt=True
                )
                ids = tokenizer.apply_chat_template(
                    conversation, truncation=True, max_length=constants.sequence_length,
                )
                batches.append((torch.stack([torch.tensor(ids)]), len(prompt_ids)))
            except Exception as e:
                bt.logging.warning(f"Error tokenizing conversation: {e}")
        return batches

    def __iter__(self):
        """
        Returns an iterator over the buffer containing collected prompts and responses.
        
        Returns:
            Iterator: An iterator over the buffer.
        """
        return iter(self.buffer)
