import bittensor as bt
from typing import Optional
from constants import CompetitionParameters, COMPETITION_SCHEDULE
import constants
import statistics
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
    def get_competition_parameters(cls, id: str) -> Optional[CompetitionParameters]:
        for x in COMPETITION_SCHEDULE:
            if x.competition_id == id:
                return x
        return None
    
    @staticmethod
    def verify_model_satisfies_parameters(model: Model) -> bool:
        parameters = ModelUpdater.get_competition_parameters(model.id.competition_id)
        if not parameters:
            bt.logging.trace(
                f"No competition parameters found for {model.id.competition_id}"
            )
            return False
        
        # Check that the parameter count of the model is within allowed bounds.
        parameter_size = sum(p.numel() for p in model.pt_model.parameters())
        if parameters.max_model_parameter_size is not None and parameter_size > parameters.max_model_parameter_size:
            return False
        
        # Check parameters are sane
        return ModelUpdater._validate_parameters(model.pt_model, 
                                   constants.norm_eps_soft,
                                   constants.norm_eps_soft_percent_threshold,
                                   constants.norm_eps_hard)


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
        
        # Backwards compatability for models submitted before competition id added
        if metadata.id.competition_id is None:
            metadata.id.competition_id = constants.ORIGINAL_COMPETITION_ID

        parameters = ModelUpdater.get_competition_parameters(metadata.id.competition_id)
        if not parameters:
            bt.logging.trace(
                f"No competition parameters found for {metadata.id.competition_id}"
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
        model = await self.remote_store.download_model(metadata.id, path, parameters)

        # Check that the hash of the downloaded content matches.
        if model.id.hash != metadata.id.hash:
            raise ValueError(
                f"Sync for hotkey {hotkey} failed. Hash of content downloaded from hugging face does not match chain metadata. {metadata}"
            )

        if not ModelUpdater.verify_model_satisfies_parameters(model):
            raise ValueError(
                    f"Sync for hotkey {hotkey} failed, model does not satisfy parameters for block {metadata.block}"
                )

        # Update the tracker
        self.model_tracker.on_miner_model_updated(hotkey, metadata)

        return True

    @staticmethod
    def _validate_parameters(base_model, eps_soft, eps_soft_percent_threshold, eps_hard) -> bool:
        """
        Validate that parameters of a model

        Parameters:
            base_model (transformers.PreTrainedModel): The base model instance.
            num_layers (int): Number of layers in the model to inspect.
            eps_soft (float): Calculate the percentage of layers above this norm
            eps_soft_percent_threshold (float): Threshold of percentage above eps_soft that will trigger a detection
            eps_hard (float): Hard limit for any norm
        """
        exceed_counts = {'k_proj': 0, 'v_proj': 0, 'up_proj': 0}
        total_counts = {'k_proj': 0, 'v_proj': 0, 'up_proj': 0}

        for layer in base_model.model.layers:
            for proj in ['k_proj', 'v_proj']:
                weight_norm = getattr(layer.self_attn, proj).weight.norm().item()
                if weight_norm > eps_hard:
                    return False
                elif weight_norm > eps_soft:
                    exceed_counts[proj] += 1
                total_counts[proj] += 1

            # up_proj is in the mlp layer
            up_proj_norm = layer.mlp.up_proj.weight.norm().item()
            if up_proj_norm > eps_hard:
                return False
            elif up_proj_norm > eps_soft:
                exceed_counts['up_proj'] += 1
            total_counts['up_proj'] += 1

        # Calculating and printing percentages
        percentages = [exceed_counts[proj] / total_counts[proj] for proj in exceed_counts]
        return statistics.fmean(percentages) <= eps_soft_percent_threshold