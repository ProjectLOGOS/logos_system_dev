from __future__ import annotations
from common.agents import SubsystemInterface
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class TetragnosAdapter(SubsystemInterface):
    config_path: str
    validator: Any

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.validator = None

    def initialize(self) -> None:
        from TETRAGNOS.ontological_validator import OntologicalValidator
        logger.debug(f'Initializing TETRAGNOS with config: {self.config_path}')
        self.validator = OntologicalValidator.load(self.config_path)

    def run(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        logger.info('Running TETRAGNOS ontology validator')
        nodes: List[Any] = knowledge.get('nodes', [])  # type: ignore
        valid: bool = self.validator.validate(nodes)
        return {'valid': valid}

    def list_commands(self) -> List[str]:
        return ['validate_nodes', 'integrate_schema']
