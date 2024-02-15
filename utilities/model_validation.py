import sys
import math
import random

from model.data import ModelId
from transformers import LlamaForCausalLM, AutoTokenizer
from finetune.dataset import CortexSubsetLoader
from finetune.validation import compute_losses
from model.data import Model
import argparse
from utilities.perf_monitor import PerfMonitor


def load_model(model_path):
    model_id = ModelId(namespace="namespace", name="name")
    pt_model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
        use_safetensors=True,
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
    args = parser.parse_args()

    print("Loading model")
    load_model_perf = PerfMonitor("Eval: Load model")
    with load_model_perf.sample():
        model = load_model(model_path=args.model_path)
    print(load_model_perf.summary_str())

    print("Getting latest Cortex data")
    pull_data_perf = PerfMonitor("Eval: Pull data")

    with pull_data_perf.sample():
        cortex_data = CortexSubsetLoader(
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
