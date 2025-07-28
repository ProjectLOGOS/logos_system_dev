import pytest
from induction.ontology_inducer import OntologyInducer

@pytest.fixture
def inducer():
    return OntologyInducer()

def test_induce_sync(inducer):
    data = [{'a': 1, 'b': 'x'}, {'a': 2, 'c': True}]
    result = inducer.induce(data)
    assert isinstance(result, dict)
    assert set(result['a']) == {'int'}
    assert set(result['b']) == {'str'}
    assert set(result['c']) == {'bool'}

def test_induce_async(inducer):
    res = inducer.induce([{'a': 3}], async_mode=True)
    assert res is True
