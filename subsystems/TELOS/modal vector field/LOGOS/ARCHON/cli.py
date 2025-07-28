import typer
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
