import requests
from core.logos_validator_hub import validator_gate

@validator_gate
def fetch_json(url: str, params: dict = None):
    """Fetch JSON data from URL."""
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()
