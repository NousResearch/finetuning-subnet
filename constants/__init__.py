from pathlib import Path
from dataclasses import dataclass
from transformers import PreTrainedModel, LlamaForCausalLM, GemmaForCausalLM, StableLmForCausalLM
from typing import Type, Optional, Any, List, Tuple
import math

@dataclass
class CompetitionParameters:
    """Class defining model parameters"""

    # The maximum parameter size allowed for models
    max_model_parameter_size: int
    # Architecture class of model
    architecture: Type[PreTrainedModel]
    # Any additional arguments to from_pretrained
    kwargs: Any
    # Fixed tokenizer
    tokenizer: Optional[str]
    # Reward percentage
    reward_percentage: float
    # Competition id
    competition_id: str

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
COMPETITION_SCHEDULE: List[CompetitionParameters] = [
    CompetitionParameters(
        max_model_parameter_size=7 * 1024 * 1024 * 1024,
        architecture=LlamaForCausalLM,
        kwargs={},
        tokenizer="mistralai/Mistral-7B-Instruct-v0.1",
        reward_percentage=0.5,
        competition_id="m1"
    ),
    CompetitionParameters(
        max_model_parameter_size=2 * 1024 * 1024 * 1024,
        architecture=StableLmForCausalLM,
        kwargs={},
        tokenizer="stabilityai/stablelm-2-zephyr-1_6b",
        reward_percentage=0.5,
        competition_id="s1"
    )
]
ORIGINAL_COMPETITION_ID = "m1"


assert math.isclose(sum(x.reward_percentage for x in COMPETITION_SCHEDULE), 1.0)
assert all(len(x.competition_id) > 0 and len(x.competition_id) <= 2 for x in COMPETITION_SCHEDULE)

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = 2002

# validator weight moving average term
alpha = 0.9
# validator scoring exponential temperature
temperature = 0.08
# validator score boosting for earlier models.
timestamp_epsilon = 0.01
# validator eval sequence length.
sequence_length = 2048

# norm validation values
norm_eps_soft = 500
norm_eps_soft_percent_threshold = 0.3
norm_eps_hard = 2500
