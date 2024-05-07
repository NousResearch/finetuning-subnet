import asyncio
import functools
import bittensor as bt
import os
from model.data import ModelId, ModelMetadata
import constants
from model.storage.model_metadata_store import ModelMetadataStore
from typing import Optional

from utilities import utils


class ChainModelMetadataStore(ModelMetadataStore):
    """Chain based implementation for storing and retrieving metadata about a model."""

    def __init__(
        self,
        subtensor: bt.subtensor,
        subnet_uid: int,
        wallet: Optional[bt.wallet] = None,
    ):
        self.subtensor = subtensor
        self.wallet = (
            wallet  # Wallet is only needed to write to the chain, not to read.
        )
        self.subnet_uid = subnet_uid

    async def store_model_metadata(self, hotkey: str, model_id: ModelId, wait_for_inclusion: bool = False, wait_for_finalization: bool = True):
        """Stores model metadata on this subnet for a specific wallet."""
        if self.wallet is None:
            raise ValueError("No wallet available to write to the chain.")
        
        data = model_id.to_compressed_str()

        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        partial = functools.partial(
            bt.extrinsics.serving.publish_metadata,
            self.wallet,
            self.subnet_uid,
            f"Raw{len(data)}",
            data.encode(),
            wait_for_inclusion,
            wait_for_finalization,
        )
        utils.run_in_subprocess(partial, 60)

    async def retrieve_model_metadata(self, hotkey: str) -> Optional[ModelMetadata]:
        """Retrieves model metadata on this subnet for specific hotkey"""

        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        partial = functools.partial(
            bt.extrinsics.serving.get_metadata, self.subtensor, self.subnet_uid, hotkey
        )

        metadata = utils.run_in_subprocess(partial, 60)

        if not metadata:
            return None

        commitment = metadata["info"]["fields"][0]
        hex_data = commitment[list(commitment.keys())[0]][2:]

        chain_str = bytes.fromhex(hex_data).decode()

        model_id = None

        try:
            model_id = ModelId.from_compressed_str(chain_str)
        except:
            # If the metadata format is not correct on the chain then we return None.
            bt.logging.trace(
                f"Failed to parse the metadata on the chain for hotkey {hotkey}."
            )
            return None

        model_metadata = ModelMetadata(id=model_id, block=metadata["block"])

        return model_metadata


# Can only commit data every ~20 minutes.
async def test_store_model_metadata():
    """Verifies that the ChainModelMetadataStore can store data on the chain."""
    model_id = ModelId(
        namespace="TestPath", name="TestModel", hash="TestHash1", commit="1.0"
    )

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor()

    # Uses .env configured wallet/hotkey/uid for the test.
    coldkey = os.getenv("TEST_COLDKEY")
    hotkey = os.getenv("TEST_HOTKEY")
    net_uid = int(os.getenv("TEST_SUBNET_UID"))

    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=wallet, subnet_uid=net_uid
    )

    # Store the metadata on chain.
    await metadata_store.store_model_metadata(hotkey=hotkey, model_id=model_id, wait_for_inclusion=True, wait_for_finalization=False)

    print(f"Finished storing {model_id} on the chain.")


async def test_retrieve_model_metadata():
    """Verifies that the ChainModelMetadataStore can retrieve data from the chain."""
    expected_model_id = ModelId(
        namespace="TestPath", name="TestModel", hash="TestHash1", commit="1.0"
    )

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor()

    # Uses .env configured hotkey/uid for the test.
    net_uid = int(os.getenv("TEST_SUBNET_UID"))
    hotkey_address = os.getenv("TEST_HOTKEY_ADDRESS")

    # Do not require a wallet for retrieving data.
    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=None, subnet_uid=net_uid
    )

    # Retrieve the metadata from the chain.
    model_metadata = await metadata_store.retrieve_model_metadata(hotkey_address)

    print(f"Expecting matching model id: {expected_model_id == model_metadata.id}")


# Can only commit data every ~20 minutes.
async def test_roundtrip_model_metadata():
    """Verifies that the ChainModelMetadataStore can roundtrip data on the chain."""
    model_id = ModelId(
        namespace="TestPath", name="TestModel", hash="TestHash1", commit="1.0"
    )

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor()

    # Uses .env configured wallet/hotkey/uid for the test.
    coldkey = os.getenv("TEST_COLDKEY")
    hotkey = os.getenv("TEST_HOTKEY")
    net_uid = int(os.getenv("TEST_SUBNET_UID"))

    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=wallet, subnet_uid=net_uid
    )

    # Store the metadata on chain.
    await metadata_store.store_model_metadata(hotkey=hotkey, model_id=model_id, wait_for_inclusion=True, wait_for_finalization=True)

    # Retrieve the metadata from the chain.
    model_metadata = await metadata_store.retrieve_model_metadata(hotkey)

    print(f"Expecting matching metadata: {model_id == model_metadata.id}")


if __name__ == "__main__":
    # Can only commit data every ~20 minutes.
    # asyncio.run(test_roundtrip_model_metadata())
    # asyncio.run(test_store_model_metadata())
    asyncio.run(test_retrieve_model_metadata())
