#!/usr/bin/env python3
import os
from zipfile import ZipFile

# Map of relative file paths → file contents
files = {
    'THONOC/adapter.py': """from __future__ import annotations
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
""",
    'TELOS/adapter.py': """from __future__ import annotations
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
""",
    'TETRAGNOS/adapter.py': """from __future__ import annotations
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
""",
    'ARCHON/unifier.py': """from __future__ import annotations
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
""",
    'ARCHON/cli.py': """import typer
import json
import sys
from typing import Any, Dict
from ARCHON.unifier import ArchonUnifier

app = typer.Typer()

@app.command()
def run(
    command: str,
    payload_file: str,
    config_file: str
) -> None:
    try:
        with open(config_file) as f:
            config: Dict[str, str] = json.load(f)
    except Exception as e:
        typer.secho(f'Error reading config: {e}', err=True)
        sys.exit(1)

    try:
        with open(payload_file) as f:
            payload: Dict[str, Any] = json.load(f)
    except Exception as e:
        typer.secho(f'Error reading payload: {e}', err=True)
        sys.exit(1)

    unifier = ArchonUnifier(config)
    unifier.initialize()

    try:
        result: Dict[str, Any] = unifier.run(command, payload)
    except ValueError as e:
        typer.secho(str(e), err=True)
        sys.exit(1)

    typer.echo(json.dumps(result, indent=2))

@app.command()
def list_commands(
    config_file: str
) -> None:
    try:
        with open(config_file) as f:
            config: Dict[str, str] = json.load(f)
    except Exception as e:
        typer.secho(f'Error reading config: {e}', err=True)
        sys.exit(1)

    unifier = ArchonUnifier(config)
    unifier.initialize()

    for cmd in unifier.list_commands():
        typer.echo(cmd)

if __name__ == '__main__':
    app()
""",
    'tests/fixtures/thonoc_expected.json': """{
  "forecast": [42, 43, 44]
}""",
    'tests/fixtures/telos_expected.json': """{
  "embedding": [0.1, 0.2, 0.3]
}""",
    'tests/fixtures/tetragnos_expected.json': """{
  "valid": true
}""",
    'tests/test_thonoc_adapter.py': """import json
from pathlib import Path
import pytest
from THONOC.adapter import ThonocAdapter

@pytest.fixture
def expected_forecast():
    path = Path(__file__).parent / "fixtures" / "thonoc_expected.json"
    return json.loads(path.read_text())

class DummyThonocEngine:
    def __init__(self, forecast):
        self._forecast = forecast
    def predict(self, ts):
        return self._forecast

def test_thonoc_run_returns_fixture(expected_forecast):
    adapter = ThonocAdapter(config_path="dummy.json")
    adapter.engine = DummyThonocEngine(expected_forecast["forecast"])
    result = adapter.run({"time_series": [1,2,3]})
    assert result == expected_forecast
""",
    'tests/test_telos_adapter.py': """import json
from pathlib import Path
import pytest
from TELOS.adapter import TelosAdapter

@pytest.fixture
def expected_embedding():
    path = Path(__file__).parent / "fixtures" / "telos_expected.json"
    return json.loads(path.read_text())

class DummyTelosGenerator:
    def __init__(self, embedding):
        self._embedding = embedding
    def generate_vector(self, point, depth):
        return self._embedding

def test_telos_run_returns_fixture(expected_embedding):
    adapter = TelosAdapter(config_path="dummy.json")
    adapter.generator = DummyTelosGenerator(expected_embedding["embedding"])
    result = adapter.run({"point": [0,0], "depth": 10})
    assert result == expected_embedding
""",
    'tests/test_tetragnos_adapter.py': """import json
from pathlib import Path
import pytest
from TETRAGNOS.adapter import TetragnosAdapter

@pytest.fixture
def expected_valid():
    path = Path(__file__).parent / "fixtures" / "tetragnos_expected.json"
    return json.loads(path.read_text())

class DummyValidator:
    def __init__(self, valid):
        self._valid = valid
    def validate(self, nodes):
        return self._valid

def test_tetragnos_run_returns_fixture(expected_valid):
    adapter = TetragnosAdapter(config_path="dummy.json")
    adapter.validator = DummyValidator(expected_valid["valid"])
    result = adapter.run({"nodes": [1,2,3]})
    assert result == expected_valid
""",
    'tests/test_cli.py': """import json
from typer.testing import CliRunner
from ARCHON.cli import app

runner = CliRunner()

def test_list_commands(tmp_path):
    config = {"THONOC":"c1.json","TELOS":"c2.json","TETRAGNOS":"c3.json"}
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(config))
    res = runner.invoke(app, ["list-commands", str(cfg)])
    assert res.exit_code == 0
    assert "predict" in res.stdout

def test_run_unknown_command(tmp_path):
    config = {"THONOC":"c1.json","TELOS":"c2.json","TETRAGNOS":"c3.json"}
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(config))
    payload = tmp_path / "payload.json"
    payload.write_text("{}")
    res = runner.invoke(app, ["run","nope",str(payload),str(cfg)])
    assert res.exit_code != 0
"""
}

# Write files
for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

# Create zip
zip_name = "archon_full_package.zip"
with ZipFile(zip_name, "w") as zipf:
    for root, dirs, fs in os.walk("."):
        for fn in fs:
            if fn == zip_name: continue
            zipf.write(os.path.join(root, fn))
print(f"Generated {zip_name} in current directory.")
