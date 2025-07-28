import pytest
from induction.skill_acquisition import SkillAcquisition

@pytest.fixture
def sa():
    return SkillAcquisition()

def test_acquire_sync(sa):
    logs = ['action:foo,param1,param2', 'log entry']
    tools = sa.acquire(logs)
    assert 'foo' in tools
    assert tools['foo']['params'] == ['param1', 'param2']

def test_acquire_async(sa):
    res = sa.acquire(['action:test'], async_mode=True)
    assert res is True
