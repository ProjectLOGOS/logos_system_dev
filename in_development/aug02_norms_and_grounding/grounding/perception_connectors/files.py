import csv
import json
from core.logos_validator_hub import validator_gate

@validator_gate
def read_csv(path: str):
    """Read a CSV file and return list of dicts."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

@validator_gate
def read_json(path: str):
    """Read a JSON file and return the parsed object."""
    with open(path) as f:
        return json.load(f)
