import abc
from typing import Optional
from model.data import ModelId, ModelMetadata


class ModelMetadataStore(abc.ABC):
    """An abstract base class for storing and retrieving model metadata."""

    @abc.abstractmethod
    async def store_model_metadata(self, hotkey: str, model_id: ModelId):
        """Stores model metadata on this subnet for a specific miner."""
        pass

    @abc.abstractmethod
    async def retrieve_model_metadata(self, hotkey: str) -> Optional[ModelMetadata]:
        """Retrieves model metadata + block information on this subnet for specific miner, if present"""
        pass
