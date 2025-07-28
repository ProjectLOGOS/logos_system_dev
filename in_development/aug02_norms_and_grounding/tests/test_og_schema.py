import pytest
from norms.og_schema import OGSchema

@pytest.fixture
def schema():
    return OGSchema()

def test_evaluate_known_actions(schema):
    score = schema.evaluate('help', context={'importance': 2.0})
    assert score == pytest.approx(2.0)
    score_default = schema.evaluate('unknown', context={})
    assert score_default == pytest.approx(0.5)
