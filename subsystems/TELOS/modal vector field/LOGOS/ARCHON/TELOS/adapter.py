from __future__ import annotations
from common.agents import SubsystemInterface
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class TelosAdapter(SubsystemInterface):
    config_path: str
    generator: Any

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.generator = None

    def initialize(self) -> None:
        from TELOS.banach_generator import BanachGenerator
        logger.debug(f'Initializing TELOS with config: {self.config_path}')
        self.generator = BanachGenerator.from_settings(self.config_path)

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        logger.info('Running TELOS fractal generator')
        point: List[float] = params.get('point', [0.0, 0.0])  # type: ignore
        depth: int = params.get('depth', 100)               # type: ignore
        embedding: List[float] = self.generator.generate_vector(point, depth)
        return {'embedding': embedding}

    def list_commands(self) -> List[str]:
        return ['generate_vector', 'visualize_fractal']
