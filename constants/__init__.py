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

__version__ = "0.2.8"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

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
        max_model_parameter_size=8 * 1024 * 1024 * 1024,
        architecture=LlamaForCausalLM,
        kwargs={},
        tokenizer="NousResearch/Meta-Llama-3-8B-Instruct",
        reward_percentage=0.6,
        competition_id="l3"
    ),
    CompetitionParameters(
        max_model_parameter_size=2 * 1024 * 1024 * 1024,
        architecture=StableLmForCausalLM,
        kwargs={},
        tokenizer="stabilityai/stablelm-2-zephyr-1_6b",
        reward_percentage=0.4,
        competition_id="s1"
    )
]
ORIGINAL_COMPETITION_ID = "m1"


assert math.isclose(sum(x.reward_percentage for x in COMPETITION_SCHEDULE), 1.0)
assert all(len(x.competition_id) > 0 and len(x.competition_id) <= 2 for x in COMPETITION_SCHEDULE)

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

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = __spec_version__

# validator weight moving average term
alpha = 0.9
# validator scoring exponential temperature
temperature = 0.08
# validator score boosting for earlier models.
timestamp_epsilon = 0.01
# validator eval sequence length.
sequence_length = 2048

# norm validation values
norm_eps_soft = 200
norm_eps_soft_percent_threshold = 0.15
norm_eps_hard = 1000
