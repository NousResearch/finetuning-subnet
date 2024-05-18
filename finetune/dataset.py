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

UNWANTED_PHRASES = [
    "text-based AI language model",
    "please refrain",
    "it is never okay",
    "It is important to",
    "It's important to",
    "real-world consequences",
    "responsible AI",
    "AI principles",
    "AI assistant",
    "an AI language",
    "as a language model",
    "as an AI language model",
    "As a large language model",
    "As an AI",
    "ethical principles",
    "it is not appropriate",
    "it's not appropriate",
    "I cannot fulfill your request",
    "ethical guidelines",
    "my guidelines",
    "prioritize user safety",
    "cannot provide guidance",
    "cannot provide information",
    "unable to offer assistance",
    "cannot engage in discussions",
    "programming prohibits",
    "follow ethical guidelines",
    "cannot support or promote",
    "against my programming",
    "not able to provide",
    "cannot provide any information",
    "an AI language model you don't have",
    "As an AI language model, I cannot",
    "As an AI language model, I do not",
    "As an AI language model, I am not able",
    "As an AI language model, I don't have personal",
    "I am an AI language model and do not",
    "However, it is important to use any code or information provided responsibly and within legal and ethical boundaries.",
    "As an AI language model, I don't have",
    "As an AI language model, I am only able",
    "AI language model and I do not",
    "As an AI language model, I cannot modify",
    "As an AI language model, I do not",
    "I know as an AI language model you don't have",
    "as an AI language model, you cannot",
    "I'm sorry, but as an AI language model",
    "As an AI language model, I don't have",
    "Unfortunately, I cannot provide",
    "I'm sorry, I cannot",
    "I'm sorry, I cannot generate",
    "AI cannot create or program",
    "I'm afraid I cannot create",
    "I cannot assist",
    "I'm sorry,",
    "I'm an AI" ,
    "I am an AI",
    "my purpose",
    "entertainment purposes",
    "purely hypothetical",
    "not a human",
    "I am an AI",
    "cannot provide",
    "can't provide",
    "won't provide",
    "not provide",
    "a language model",
    "As a machine",
    "I don't have the ability",
    "I am here to assist",
    "my purpose is to ",
    "my knowledge cutoff",
    "my knowledge cut off",
    "September 2021",
    "I apologize, but",
    "It is not possible",
    "Please note",
    "not acceptable",
    "*This chat conversation is shared from",
    "*This conversation is shared from",
    "<|endoftext|>",
    "Я разработчик",
    "I'm sorry, I cannot",
    "breach of",
    "privacy policy",
    "I am programmed to",
    "As a helpful assistant",
    "I don't have beliefs",
    "I don't have personal",
    "I don't have a personal",
    "I don't have emotions",
    "I don't have the ability to feel",
    "I don't have a physical",
    "I don't have physical",
    "I don't have the ability to remember",
    "I don't have access to real-time",
    "I don't have sensors or a physical body",
    "I don't have sensory input",
    "I don't have a sense",
    "I don't have the capability to perceive",
    "I don't have the capability to feel",
    "I am an artificial intelligence",
    "I don't have access to real-time",
    "I don't have beliefs or disagreements",
    "I do not have a sense of",
    "I do not have beliefs",
    "I do not have personal",
    "I do not have a personal",
    "I do not have emotions",
    "I do not have the ability to feel",
    "I do not have a physical",
    "I do not have physical",
    "I do not have the ability to remember",
    "I do not have access to real-time",
    "I do not have sensors or a physical body",
    "I do not have sensory input",
    "I do not have a sense",
    "I do not have the capability to perceive",
    "I do not have the capability to feel",
    "I am an artificial intelligence",
    "I do not have access to real-time",
    "I do not have beliefs or disagreements",
    "I do not have a sense of",
    "September 2021",
    "as a language model",
    "ethical guidelines",
    "as an AI language model",
    "my guidelines",
    "As an AI",
    "cannot provide guidance",
    "cannot provide information",
    "unable to offer assistance",
    "cannot engage in discussions",
    "programming prohibits",
    "cannot support or promote",
    "activities that could harm",
    "against my programming",
    "activities that could undermine",
    "not within the scope",
    "designed to prioritize safety",
    "not able to provide",
    "maintain user safety",
    "adhere to safety guidelines",
    "dangerous or harmful",
    "cannot provide any information",
    "focus on promoting safety",
    "maintain user safety",
    "focus on promoting safety",
    "it is never okay",
    "September 2021",
    "as a language model",
    "ethical guidelines",
    "as an AI language model",
    "my guidelines",
    "As an AI",
    "prioritize user safety",
    "adhere to ethical guidelines",
    "promote safety",
    "responsible information sharing",
    "jeopardize the safety",
    "safe information",
    "cannot provide guidance",
    "cannot provide information",
    "unable to offer assistance",
    "cannot engage in discussions",
    "programming prohibits",
    "prioritize safety",
    "cannot support or promote",
    "activities that could harm",
    "against my programming",
    "potentially dangerous",
    "not within the scope",
    "not able to provide",
    "cannot provide any information",
    "I don't have beliefs"
    "I don't have personal"
    "gpt",
    "gpT",
    "gPt",
    "Gpt",
    "gPT",
    "GpT",
    "GPt",
    "GPT",
    "gpt"
]

class CortexSubsetLoader(IterableDataset):
    def __init__(
        self, 
        subtensor: typing.Optional[bt.subtensor] = None,
        latest=True,
        random_seed: typing.Optional[int] = None,
        max_samples=300,
        steps: typing.Optional[int]=1,
        progress=False,
        retry_limit=10, 
        page_size=100, 
        running: typing.Optional[bool]=False,
        cortex_project=constants.CORTEX_WANDB_PROJECT,
        cortex_type=constants.CORTEX_WANDB_TYPE):
        api = wandb.Api(timeout=100)

        filters = [{ "config.type": cortex_type }]
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
                    if run.config:
                        id = run.id
                        hotkey = run.config.get("hotkey")
                        signature = run.config.get("signature")
                        if id and hotkey and signature:
                            keypair = Keypair(ss58_address=hotkey)
                            verified = keypair.verify(id.encode(), bytes.fromhex(signature))
                            if verified and subtensor is not None:
                                stake = subtensor.get_total_stake_for_hotkey(hotkey)
                                stake_int = int(stake)
                                if stake_int > 25000000000000:
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
                            if sample and sample["modality"] == "text":
                                for uid in range(constants.CORTEX_MAX_UIDS):
                                    try:
                                        prompt: typing.Optional[str] = sample[f"prompts.{uid}"]
                                        response: typing.Optional[str]  = sample[f"responses.{uid}"]
                                        score: typing.Optional[float] = sample[f"scores.{uid}"]
                                        if isinstance(prompt, str) and isinstance(response, str) and isinstance(score, float):
                                            prompt = prompt.strip()
                                            response = response.strip()
                                            if len(prompt) > 0 and len(response) > 0 and score > 0.0:
                                                if not any(x in response for x in UNWANTED_PHRASES):
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
