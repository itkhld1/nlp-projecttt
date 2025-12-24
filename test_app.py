import pytest
from app import load_model
import config

# just checking that the model actually exists where we expect it to
# if training hasn't been done yet, these tests will probably fail

def test_load_model():
    """
    making sure the model loads without blowing up
    """
    classifier = load_model()
    assert classifier is not None, "model load failed. should not be None"

def test_classification_output_format():
    """
    testing if the prediction output looks like the format we expect
    """
    classifier = load_model()
    if classifier is None:
        pytest.fail("couldn't load model, so we can't test the classifier :/")

    sample_text = "Starbucks Coffee"
    result = classifier(sample_text)

    # kinda sanity-checking everything here
    assert isinstance(result, list), "expected a list back from classifier"
    assert len(result) > 0, "list shouldn't be empty"
    assert isinstance(result[0], dict), "each prediction should be a dict"
    assert "label" in result[0], "missing 'label' field"
    assert "score" in result[0], "missing 'score' field"
    assert isinstance(result[0]["label"], str), "label should be a string"
    assert isinstance(result[0]["score"], float), "score should be a float"
    assert 0.0 <= result[0]["score"] <= 1.0, "score must be between 0 and 1"

def test_classification_with_multiple_inputs():
    """
    checking that the classifier can handle multiple items at once
    """
    classifier = load_model()
    if classifier is None:
        pytest.fail("model didn't load, so multi-input test can't run")

    sample_texts = ["Amazon purchase", "Uber ride", "Netflix subscription"]
    results = classifier(sample_texts)  # pipeline supports list inputs

    assert isinstance(results, list)
    assert len(results) == len(sample_texts)

    # looping through predictions just to be extra sure
    for prediction in results:
        assert isinstance(prediction, dict)
        assert "label" in prediction
        assert "score" in prediction
        assert isinstance(prediction["label"], str)
        assert isinstance(prediction["score"], float)
        assert 0.0 <= prediction["score"] <= 1.0
