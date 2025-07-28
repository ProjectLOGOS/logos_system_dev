import requests
from core.logos_validator_hub import validator_gate

class APIClient:
    """
    Generic API client for authenticated endpoints.
    """
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}

    @validator_gate
    def get(self, path: str, params: dict = None):
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = requests.get(url, headers=self.headers, params=params)
        resp.raise_for_status()
        return resp.json()

    @validator_gate
    def post(self, path: str, payload: dict):
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = requests.post(url, headers=self.headers, json=payload)
        resp.raise_for_status()
        return resp.json()
