import sys
import math
import random

import bittensor as bt
from model.data import ModelId
from transformers import AutoTokenizer
from finetune.dataset import CortexSubsetLoader
from finetune.validation import compute_losses
from model.data import Model
import argparse
import constants
import torch
from model.model_updater import ModelUpdater
from utilities.perf_monitor import PerfMonitor


def load_model(model_path, parameters: constants.CompetitionParameters):
    model_id = ModelId(namespace="namespace", name="name", competition_id=parameters.competition_id)
    pt_model = parameters.architecture.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
        use_safetensors=True,
        **parameters.kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
    )
    return Model(id=model_id, pt_model=pt_model, tokenizer=tokenizer)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, help="Local path to your model", required=True
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device name.",
    )
    parser.add_argument(
        "--latest_cortex_steps",
        type=int,
        default=1,
        help="Number of most recent Cortex steps to sample data from",
    )
    parser.add_argument(
        "--latest_cortex_samples",
        type=int,
        default=400,
        help="Number of most recent Cortex samples to eval against",
    )
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        help="Implementation of attention to use",
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
        help="competition to validate against (use --list-competitions to get all competitions)"
    )
    parser.add_argument(
        "--list_competitions",
        action="store_true",
        help="Print out all competitions"
    )
    args = parser.parse_args()
    if args.list_competitions:
        print(constants.COMPETITION_SCHEDULE)
        return
    
    competition_parameters = ModelUpdater.get_competition_parameters(args.competition_id)
    competition_parameters.kwargs["torch_dtype"] = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    competition_parameters.kwargs["attn_implementation"] = args.attn_implementation
    competition_parameters.kwargs["use_cache"] = True

    print(f"Loading model for competition {args.competition_id}")
    load_model_perf = PerfMonitor("Eval: Load model")
    with load_model_perf.sample():
        model = load_model(args.model_path, competition_parameters)
    print(load_model_perf.summary_str())

    if not ModelUpdater.verify_model_satisfies_parameters(model):
        print("Model does not satisfy competition parameters!!!")
        return

    print("Getting latest Cortex data")
    pull_data_perf = PerfMonitor("Eval: Pull data")

    with pull_data_perf.sample():
        cortex_data = CortexSubsetLoader(
            subtensor=bt.subtensor(),
            latest=True,
            running=True,
            random_seed=random.randint(0, sys.maxsize),
            max_samples=args.latest_cortex_samples,
            steps=args.latest_cortex_steps,
            page_size=args.latest_cortex_steps,
        )
    print(pull_data_perf.summary_str())

    print("Tokenizing cortex data")
    tokenizer = model.tokenizer
    batches = cortex_data.tokenize(tokenizer)

    print("Calculating losses")
    compute_loss_perf = PerfMonitor("Eval: Compute loss")
    with compute_loss_perf.sample():
        losses = compute_losses(model.pt_model, batches, device=args.device)
    print(compute_loss_perf.summary_str())

    average_model_loss = sum(losses) / len(losses) if len(losses) > 0 else math.inf

    print(f"The average model loss for {args.model_path} is {average_model_loss}")


if __name__ == "__main__":
    main()
