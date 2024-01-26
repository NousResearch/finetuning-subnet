from model.data import Model, ModelId
from model.storage.disk import utils
from model.storage.remote_model_store import RemoteModelStore


class FakeRemoteModelStore(RemoteModelStore):
    """Fake implementation for remotely storing and retrieving a model."""

    def __init__(self):
        self.remote_models = dict()

    async def upload_model(self, model: Model) -> ModelId:
        """Fake uploads a model."""

        # Use provided commit + hash rather than generating a new one.
        self.remote_models[model.id] = model

        return model.id

    async def download_model(self, model_id: ModelId, local_path: str) -> Model:
        """Retrieves a trained model from memory."""

        model = self.remote_models[model_id]

        # Parse out the hotkey and the base path from local_path to replicate hugging face logic.
        split_string = local_path.split("/")

        # Store it at the local_path
        model.pt_model.save_pretrained(
            save_directory=utils.get_local_model_snapshot_dir(
                split_string[0], split_string[2], model_id
            ),
            safe_serialization=True,
        )

        return model

    def inject_mismatched_model(self, model_id: ModelId, model: Model) -> ModelId:
        """Fake uploads a model by a specific model id."""

        # Use provided commit + hash rather than generating a new one.
        self.remote_models[model_id] = model

        return model_id

    def get_only_model(self) -> Model:
        """Returns the only uploaded model or raises a ValueError if none or more than one is found."""

        if len(self.remote_models) != 1:
            raise ValueError(
                f"There are {len(self.remote_models)} uploaded models. Expected 1."
            )

        return [i for i in self.remote_models.values()][0]

    def reset(self):
        """Resets the store to its initial state."""
        self.remote_models = dict()
