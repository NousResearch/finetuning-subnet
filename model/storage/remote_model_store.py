import abc
from model.data import Model, ModelId
from constants import CompetitionParameters
from typing import Optional

class RemoteModelStore(abc.ABC):
    """An abstract base class for storing and retrieving a pre trained model."""

    @abc.abstractmethod
    async def upload_model(self, model: Model, parameters: CompetitionParameters) -> ModelId:
        """Uploads a trained model in the appropriate location based on implementation."""
        pass

    @abc.abstractmethod
    async def download_model(self, model_id: ModelId, local_path: str, parameters: CompetitionParameters) -> Model:
        """Retrieves a trained model from the appropriate location and stores at the given path."""
        pass
