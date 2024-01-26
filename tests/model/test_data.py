import unittest
from model.data import ModelId
import model.storage.disk.utils as utils
import os


class TestData(unittest.TestCase):
    def test_model_id_compressed_string(self):
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            commit="test_commit",
            hash="test_hash",
        )

        roundtrip_model_id = ModelId.from_compressed_str(model_id.to_compressed_str())

        self.assertEqual(model_id, roundtrip_model_id)

    def test_model_id_compressed_string_no_commit(self):
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            hash="test_hash",
        )

        roundtrip_model_id = ModelId.from_compressed_str(model_id.to_compressed_str())

        self.assertEqual(model_id, roundtrip_model_id)


if __name__ == "__main__":
    unittest.main()
