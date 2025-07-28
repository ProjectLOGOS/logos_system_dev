from __future__ import annotations
from common.agents import SubsystemInterface
from THONOC.adapter import ThonocAdapter
from TELOS.adapter import TelosAdapter
from TETRAGNOS.adapter import TetragnosAdapter
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class ArchonUnifier(SubsystemInterface):
    config: Dict[str, str]
    adapters: List[SubsystemInterface]

    def __init__(self, config: Dict[str, str]) -> None:
        self.config = config
        self.adapters = []

    def initialize(self) -> None:
        self.adapters = [
            ThonocAdapter(self.config['THONOC']),
            TelosAdapter(self.config['TELOS']),
            TetragnosAdapter(self.config['TETRAGNOS']),
        ]
        for adapter in self.adapters:
            adapter.initialize()

    def run(self, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        for adapter in self.adapters:
            if command in adapter.list_commands():
                logger.info(f'Dispatching {command} to {adapter.__class__.__name__}')
                return adapter.run(payload)
        raise ValueError(f'Unknown command: {command}')

    def list_commands(self) -> List[str]:
        commands: List[str] = []
        for adapter in self.adapters:
            commands.extend(adapter.list_commands())
        return commands
