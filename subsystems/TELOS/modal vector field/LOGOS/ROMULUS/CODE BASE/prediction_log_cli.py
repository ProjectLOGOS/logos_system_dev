import json

def load_predictions(path="prediction_log.jsonl") -> list:
    """Load all prediction logs from a file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def summarize_predictions(predictions: list) -> dict:
    """Return stats on modal outcomes, coherence ranges, etc."""
    summary = {
        "total": len(predictions),
        "modal_counts": {},
        "coherence_avg": 0.0,
        "deepest_fractal": None
    }

    total_coherence = 0.0
    max_depth = 0
    deepest = None

    for p in predictions:
        status = p["modal_status"]
        summary["modal_counts"][status] = summary["modal_counts"].get(status, 0) + 1

        coh = p["coherence"]
        total_coherence += coh

        depth = p["fractal"]["iterations"]
        if depth > max_depth:
            max_depth = depth
            deepest = p

    summary["coherence_avg"] = round(total_coherence / len(predictions), 3)
    summary["deepest_fractal"] = {
        "query": deepest.get("comment"),
        "depth": max_depth,
        "trinity": deepest["trinity"]
    } if deepest else None

    return summary
