from typing import Any, ClassVar, Dict, Optional, Type
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from pydantic import BaseModel, Field, PositiveInt

# The maximum bytes for metadata on the chain.
MAX_METADATA_BYTES = 128
# The length, in bytes, of a git commit hash.
GIT_COMMIT_LENGTH = 40
# The length, in bytes, of a base64 encoded sha256 hash.
SHA256_BASE_64_LENGTH = 44
# The max length, in characters, of the competition id
MAX_COMPETITION_ID_LENGTH = 2

class ModelId(BaseModel):
    """Uniquely identifies a trained model"""

    MAX_REPO_ID_LENGTH: ClassVar[int] = (
        MAX_METADATA_BYTES - GIT_COMMIT_LENGTH - SHA256_BASE_64_LENGTH - MAX_COMPETITION_ID_LENGTH - 4  # separators
    )

    namespace: str = Field(
        description="Namespace where the model can be found. ex. Hugging Face username/org."
    )
    name: str = Field(description="Name of the model.")

    # When handling a model locally the commit and hash are not necessary.
    # Commit must be filled when trying to download from a remote store.
    commit: Optional[str] = Field(
        description="Commit of the model. May be empty if not yet committed."
    )
    # Hash is filled automatically when uploading to or downloading from a remote store.
    hash: Optional[str] = Field(description="Hash of the trained model.")
    # Identifier for competition
    competition_id: Optional[str] = Field(description="The competition id")

    def to_compressed_str(self) -> str:
        """Returns a compressed string representation."""
        return f"{self.namespace}:{self.name}:{self.commit}:{self.hash}:{self.competition_id}"

    @classmethod
    def from_compressed_str(cls, cs: str) -> Type["ModelId"]:
        """Returns an instance of this class from a compressed string representation"""
        tokens = cs.split(":")
        return cls(
            namespace=tokens[0],
            name=tokens[1],
            commit=tokens[2] if tokens[2] != "None" else None,
            hash=tokens[3] if tokens[3] != "None" else None,
            competition_id=tokens[4] if len(tokens) >= 5 and tokens[4] != "None" else None
        )


class Model(BaseModel):
    """Represents a pre trained foundation model."""

    class Config:
        arbitrary_types_allowed = True

    id: ModelId = Field(description="Identifier for this model.")
    # PreTrainedModel.base_model returns torch.nn.Module if needed.
    pt_model: PreTrainedModel = Field(description="Pre trained model.")
    tokenizer: PreTrainedTokenizerBase = Field(description="Pre trained tokenizer.")


class ModelMetadata(BaseModel):
    id: ModelId = Field(description="Identifier for this trained model.")
    block: PositiveInt = Field(
        description="Block on which this model was claimed on the chain."
    )
