# seed_data_nodes.py
"""
Populate the Divine Plane with raw data seed nodes from uploaded files:
- Universe Success Criterion (from UNIVERSE_CRITERION.py)
- Universe Metrics (from UNIVERSE_METRICS.json)
- Bayesian Priors (from bayes_priors.json)
- Any other JSON data found in /mnt/data
"""
import os
import json
import runpy

# --- 1. Define DataSeedNode class ---
class DataSeedNode:
    def __init__(self, name: str, payload: any):
        self.name = name
        self.payload = payload
        self.links = []  # Can be linked to ontological nodes if desired

    def __repr__(self):
        return f"<DataSeedNode {self.name}: {type(self.payload).__name__}>"

# --- 2. Utility to load JSON files ---
def load_json(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- 3. Load and create seed nodes ---
seed_nodes = []

# 3a. Universe Metrics
metrics_path = '/mnt/data/UNIVERSE_METRICS.json'
if os.path.exists(metrics_path):
    metrics = load_json(metrics_path)
    seed_nodes.append(DataSeedNode('UniverseMetrics', metrics))

# 3b. Bayesian Priors
priors_path = '/mnt/data/bayes_priors.json'
if os.path.exists(priors_path):
    priors = load_json(priors_path)
    seed_nodes.append(DataSeedNode('BayesPriors', priors))

# 3c. Universe Success Criterion (execute python file)
criterion_path = '/mnt/data/UNIVERSE_CRITERION.py'
if os.path.exists(criterion_path):
    crit_namespace = runpy.run_path(criterion_path)
    # Collect any dict or list variables
    for key, val in crit_namespace.items():
        if isinstance(val, (dict, list)):
            seed_nodes.append(DataSeedNode(key, val))

# 3d. Any other .json files (excluding ontological dicts)
excluded = {'ONTOPROP_DICT.json', 'CONNECTIONS.json', 'final_ontological_property_dict.json'}
for fname in os.listdir('/mnt/data'):
    if fname.lower().endswith('.json') and fname not in excluded:
        path = os.path.join('/mnt/data', fname)
        try:
            content = load_json(path)
            names = [node.name for node in seed_nodes]
            name = os.path.splitext(fname)[0]
            if name not in names:
                seed_nodes.append(DataSeedNode(name, content))
        except Exception:
            continue

# --- 4. Quick test output ---
print(f"Populated {len(seed_nodes)} raw data seed nodes:")
for node in seed_nodes:
    print(f" - {node.name}: {type(node.payload).__name__}")
