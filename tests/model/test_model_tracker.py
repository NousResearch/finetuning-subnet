import os
import unittest
from model.data import ModelId, ModelMetadata

from model.model_tracker import ModelTracker


class TestModelTracker(unittest.TestCase):
    def setUp(self):
        self.model_tracker = ModelTracker()

    def test_roundtrip_state(self):
        hotkey = "test_hotkey"
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            commit="test_commit",
            hash="test_hash",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        state_path = ".test_tracker_state.pickle"
        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)
        self.model_tracker.save_state(state_path)

        new_tracker = ModelTracker()
        new_tracker.load_state(state_path)

        os.remove(state_path)

        self.assertEqual(
            self.model_tracker.miner_hotkey_to_model_metadata_dict,
            new_tracker.miner_hotkey_to_model_metadata_dict,
        )

    def test_on_miner_model_updated_add(self):
        hotkey = "test_hotkey"
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            commit="test_commit",
            hash="test_hash",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)

        self.assertTrue(
            hotkey in self.model_tracker.miner_hotkey_to_model_metadata_dict
        )
        self.assertEqual(
            model_metadata,
            self.model_tracker.miner_hotkey_to_model_metadata_dict[hotkey],
        )

    def test_on_miner_model_updated_update(self):
        hotkey = "test_hotkey"
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            commit="test_commit",
            hash="test_hash",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        new_model_id = ModelId(
            namespace="test_model2",
            name="test_name2",
            commit="test_commit2",
            hash="test_hash2",
        )
        new_model_metadata = ModelMetadata(id=new_model_id, block=2)

        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)
        self.model_tracker.on_miner_model_updated(hotkey, new_model_metadata)

        self.assertTrue(
            hotkey in self.model_tracker.miner_hotkey_to_model_metadata_dict
        )
        self.assertEqual(
            new_model_metadata,
            self.model_tracker.miner_hotkey_to_model_metadata_dict[hotkey],
        )

    def test_get_model_metadata_for_miner_hotkey(self):
        hotkey = "test_hotkey"
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            commit="test_commit",
            hash="test_hash",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)
        returned_model_metadata = (
            self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
        )

        self.assertEqual(model_metadata, returned_model_metadata)

    def test_get_model_metadata_for_miner_hotkey_optional(self):
        hotkey = "test_hotkey"

        returned_model_id = self.model_tracker.get_model_metadata_for_miner_hotkey(
            hotkey
        )

        self.assertIsNone(returned_model_id)

    def test_get_miner_hotkey_to_model_metadata_dict(self):
        hotkey_1 = "test_hotkey"
        model_id_1 = ModelId(
            namespace="test_model",
            name="test_name",
            commit="test_commit",
            hash="test_hash",
        )
        model_metadata_1 = ModelMetadata(id=model_id_1, block=1)

        hotkey_2 = "test_hotkey2"
        model_id_2 = ModelId(
            namespace="test_model2",
            name="test_name2",
            commit="test_commit2",
            hash="test_hash2",
        )
        model_metadata_2 = ModelMetadata(id=model_id_2, block=2)

        self.model_tracker.on_miner_model_updated(hotkey_1, model_metadata_1)
        self.model_tracker.on_miner_model_updated(hotkey_2, model_metadata_2)

        hotkey_to_model_metadata = (
            self.model_tracker.get_miner_hotkey_to_model_metadata_dict()
        )

        self.assertEqual(len(hotkey_to_model_metadata), 2)
        self.assertEqual(hotkey_to_model_metadata[hotkey_1], model_metadata_1)
        self.assertEqual(hotkey_to_model_metadata[hotkey_2], model_metadata_2)

    def test_on_hotkeys_updated_extra_ignored(self):
        hotkey = "test_hotkey"
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            commit="test_commit",
            hash="test_hash",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)
        self.model_tracker.on_hotkeys_updated(set([hotkey, "extra_hotkey"]))

        self.assertEqual(len(self.model_tracker.miner_hotkey_to_model_metadata_dict), 1)

    def test_on_hotkeys_updated_missing_removed(self):
        hotkey = "test_hotkey"
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            commit="test_commit",
            hash="test_hash",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)
        self.model_tracker.on_hotkeys_updated(set(["extra_hotkey"]))

        self.assertEqual(len(self.model_tracker.miner_hotkey_to_model_metadata_dict), 0)


if __name__ == "__main__":
    unittest.main()
