from core.logos_validator_hub import validator_gate

@validator_gate
def normalize_sensor_data(data: dict, schema: dict):
    """Normalize raw sensor readings to a standard structure."""
    normalized = {}
    for key, (min_val, max_val) in schema.items():
        raw = data.get(key, 0)
        normalized[key] = (raw - min_val) / (max_val - min_val) if max_val > min_val else 0
    return normalized

@validator_gate
def extract_metrics(api_data: dict, fields: list):
    """Extract specified fields from API data for measurement queries."""
    return {f: api_data.get(f) for f in fields}
