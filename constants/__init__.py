from pathlib import Path
from dataclasses import dataclass
from transformers import PreTrainedModel, LlamaForCausalLM
from typing import Type, Optional, Any, List, Tuple
import torch

@dataclass
class ModelParameters:
    """Class defining model parameters"""

    # The maximum parameter size allowed for models
    max_model_parameter_size: int
    # Architecture class of model
    architecture: Type[PreTrainedModel]
    # Any additional arguments to from_pretrained
    kwargs: Any
    # Fixed tokenizer
    tokenizer: Optional[str]

# ---------------------------------
# Project Constants.
# ---------------------------------

# The validator WANDB project.
WANDB_PROJECT = "finetuning-subnet"
# The uid for this subnet.
SUBNET_UID = 6
# The start block of this subnet
SUBNET_START_BLOCK = 2225782
# The Cortex.t validator WANDB project and filters
CORTEX_WANDB_PROJECT = "cortex-t/multi-modality"
CORTEX_WANDB_TYPE = "validator"
CORTEX_MAX_UIDS = 256
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent
# The maximum bytes for the hugging face repo
MAX_HUGGING_FACE_BYTES: int = 15 * 1024 * 1024 * 1024
# Schedule of model architectures
MODEL_PARAMETER_SCHEDULE: List[Tuple[int, ModelParameters]] = [
    (0, ModelParameters(
            max_model_parameter_size=7 * 1024 * 1024 * 1024,
            architecture=LlamaForCausalLM,
            kwargs={},
            tokenizer="mistralai/Mistral-7B-Instruct-v0.1"
        )
    )
]

assert \
    all(
        MODEL_PARAMETER_SCHEDULE[i][0] < MODEL_PARAMETER_SCHEDULE[i+1][0] \
            for i in range(len(MODEL_PARAMETER_SCHEDULE) - 1)
    )

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = 2002

# validator weight moving average term
alpha = 0.95
# validator scoring exponential temperature
temperature = 0.08
# validator score boosting for earlier models.
timestamp_epsilon = 0.01
# validator eval sequence length.
sequence_length = 2048
