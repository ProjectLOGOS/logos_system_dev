import pytest

MODULES = [
    'causal.scm',
    'causal.intervene',
    'causal.counterfactuals',
    'agency.planner',
    'core.async_workers',
    'core.logos_validator_hub',
    'core.config_loader'
]

@pytest.mark.parametrize('module', MODULES)
def test_import(module):
    __import__(f'logos.{module}')
