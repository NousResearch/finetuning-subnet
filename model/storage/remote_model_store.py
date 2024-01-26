import abc
from model.data import Model, ModelId
from constants import ModelParameters
from typing import Optional

class RemoteModelStore(abc.ABC):
    """An abstract base class for storing and retrieving a pre trained model."""

    @abc.abstractmethod
    async def upload_model(self, model: Model, parameters: ModelParameters) -> ModelId:
        """Uploads a trained model in the appropriate location based on implementation."""
        pass

    @abc.abstractmethod
    async def download_model(self, model_id: ModelId, local_path: str, parameters: ModelParameters) -> Model:
        """Retrieves a trained model from the appropriate location and stores at the given path."""
        pass
