from __future__ import annotations
from common.agents import SubsystemInterface
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class ThonocAdapter(SubsystemInterface):
    config_path: str
    engine: Any

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.engine = None

    def initialize(self) -> None:
        from THONOC.core.prediction_engine import PredictionEngine
        logger.debug(f'Initializing THONOC with config: {self.config_path}')
        self.engine = PredictionEngine.load_from_json(self.config_path)

    def run(self, data: Dict[str, List[float]]) -> Dict[str, Any]:
        logger.info('Running THONOC forecasting engine')
        ts = data.get('time_series', [])
        forecast: List[float] = self.engine.predict(ts)
        return {'forecast': forecast}

    def list_commands(self) -> List[str]:
        return ['predict', 'update_priors']
