import unittest
from model.utils import get_model_criteria, get_hash_of_two_strings
from constants import BLOCK_7B, ALLOWED_MODEL_TYPES_1, ALLOWED_MODEL_TYPES_2
from model.data import ModelCriteria, TokenizerIdentifier


class TestModelUtils(unittest.TestCase):
    def test_get_hash_of_two_strings(self):
        string1 = "hello"
        string2 = "world"

        result = get_hash_of_two_strings(string1, string2)

        self.assertEqual(result, "k2oYXKqiZrucvpgengXLeM1zKwsygOuURBK7b4+PB68=")


if __name__ == "__main__":
    unittest.main()