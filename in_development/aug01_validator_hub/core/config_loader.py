import yaml
from pathlib import Path

class Config:
    """
    Loads performance policy and other settings from YAML config.
    Usage:
        cfg = Config()
        perf = cfg.perf_policy
    """
    def __init__(self, config_dir: str = None):
        base = Path(config_dir or Path(__file__).parent.parent / 'config')
        policy_file = base / 'perf_policy.yaml'
        if not policy_file.exists():
            raise FileNotFoundError(f"Perf policy not found: {policy_file}")
        with open(policy_file, 'r') as f:
            data = yaml.safe_load(f)
        self._data = data

    @property
    def perf_policy(self):
        """Return performance caps for all tiers."""
        return self._data.get('tiers', {})

    def get_tier(self, name: str):
        """Return settings dict for a specific tier (interactive, standard, deep)."""
        return self.perf_policy.get(name, {})
