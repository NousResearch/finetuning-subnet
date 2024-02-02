import bittensor as bt
from typing import Optional
from constants import ModelParameters, MODEL_PARAMETER_SCHEDULE
from model.data import ModelMetadata, Model
from model.model_tracker import ModelTracker
from model.storage.local_model_store import LocalModelStore
from model.storage.model_metadata_store import ModelMetadataStore
from model.storage.remote_model_store import RemoteModelStore


class ModelUpdater:
    """Checks if the currently tracked model for a hotkey matches what the miner committed to the chain."""

    def __init__(
        self,
        metadata_store: ModelMetadataStore,
        remote_store: RemoteModelStore,
        local_store: LocalModelStore,
        model_tracker: ModelTracker
    ):
        self.metadata_store = metadata_store
        self.remote_store = remote_store
        self.local_store = local_store
        self.model_tracker = model_tracker
        self.min_block: Optional[int] = None

    def set_min_block(self, val: Optional[int]):
        self.min_block = val

    @classmethod
    def get_model_parameters_for_block(cls, block: int) -> Optional[ModelParameters]:
        for i in reversed(range(0, len(MODEL_PARAMETER_SCHEDULE))):
            schedule_block, parameters = MODEL_PARAMETER_SCHEDULE[i]
            if block >= schedule_block:
                return parameters
        return None
    
    def verify_model_satisfies_parameters(block: int, model: Model) -> bool:
        model_parameters = ModelUpdater.get_model_parameters_for_block(block)
        if not model_parameters:
            bt.logging.trace(
                f"No model parameters found for block {block}"
            )
            return False
        
        # Check that the parameter count of the model is within allowed bounds.
        parameter_size = sum(p.numel() for p in model.pt_model.parameters())
        if model_parameters.max_model_parameter_size is not None and parameter_size > model_parameters.max_model_parameter_size:
            return False
        
        return True


    async def _get_metadata(self, hotkey: str) -> Optional[ModelMetadata]:
        """Get metadata about a model by hotkey"""
        return await self.metadata_store.retrieve_model_metadata(hotkey)

    async def sync_model(self, hotkey: str) -> bool:
        """Updates local model for a hotkey if out of sync and returns if it was updated."""
        # Get the metadata for the miner.
        metadata = await self._get_metadata(hotkey)

        if not metadata:
            bt.logging.trace(
                f"No valid metadata found on the chain for hotkey {hotkey}"
            )
            return False

        if self.min_block and metadata.block < self.min_block:
            bt.logging.trace(
                f"Skipping model for {hotkey} since it was submitted at block {metadata.block} which is less than the minimum block {self.min_block}"
            )
            return False

        model_parameters = ModelUpdater.get_model_parameters_for_block(metadata.block)
        if not model_parameters:
            bt.logging.trace(
                f"No model parameters found for block {metadata.block}"
            )
            return False

        # Check what model id the model tracker currently has for this hotkey.
        tracker_model_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
            hotkey
        )
        if metadata == tracker_model_metadata:
            return False

        # Get the local path based on the local store to download to (top level hotkey path)
        path = self.local_store.get_path(hotkey)

        # Otherwise we need to download the new model based on the metadata.
        model = await self.remote_store.download_model(metadata.id, path, model_parameters)

        # Check that the hash of the downloaded content matches.
        if model.id.hash != metadata.id.hash:
            raise ValueError(
                f"Sync for hotkey {hotkey} failed. Hash of content downloaded from hugging face does not match chain metadata. {metadata}"
            )

        if not ModelUpdater.verify_model_satisfies_parameters(metadata.block, model):
            raise ValueError(
                    f"Sync for hotkey {hotkey} failed, model does not satisfy parameters for block {metadata.block}"
                )

        # Update the tracker
        self.model_tracker.on_miner_model_updated(hotkey, metadata)

        return True
