from collections import deque
from typing import List, Optional
from model.data import ModelId, ModelMetadata
from model.storage.model_metadata_store import ModelMetadataStore


class FakeModelMetadataStore(ModelMetadataStore):
    """Fake implementation for storing and retrieving metadata about a model."""

    def __init__(self):
        self.current_block = 1
        self.metadata = dict()
        self.store_errors = deque()

    async def store_model_metadata(self, hotkey: str, model_id: ModelId):
        """Fake stores model metadata for a specific hotkey."""

        # Return an injected error if we have one.
        if len(self.store_errors) > 0:
            raise self.store_errors.popleft()

        model_metadata = ModelMetadata(id=model_id, block=self.current_block)
        self.current_block += 1

        self.metadata[hotkey] = model_metadata

    async def store_model_metadata_exact(
        self, hotkey: str, model_metadata: ModelMetadata
    ):
        """Fake stores model metadata for a specific hotkey."""

        # Return an injected error if we have one.
        if len(self.store_errors) > 0:
            raise self.store_errors.popleft()

        self.metadata[hotkey] = model_metadata

    async def retrieve_model_metadata(self, hotkey: str) -> Optional[ModelMetadata]:
        """Retrieves model metadata for a specific hotkey"""

        return self.metadata[hotkey] if hotkey in self.metadata else None

    def inject_store_errors(self, errors: List[Exception]):
        """Injects a list of errors to be raised on the next N calls to store_model_metadata."""
        self.store_errors.extend(errors)

    def reset(self):
        """Resets the store to its initial state."""
        self.current_block = 1
        self.metadata = dict()
        self.store_errors = deque()
