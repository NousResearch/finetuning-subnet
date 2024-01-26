import unittest
from transformers import PreTrainedModel


def assert_model_equality(
    test_case: unittest.TestCase, model1: PreTrainedModel, model2: PreTrainedModel
):
    """Checks if two models are equal."""
    test_case.assertEqual(type(model1), type(model2))
    test_case.assertEqual(str(model1.state_dict()), str(model2.state_dict()))
