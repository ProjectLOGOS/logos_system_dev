from causal.scm import SCM

def evaluate_counterfactual(scm: SCM, target: str, context: dict, intervention: dict):
    """High-level API: P(target | do(intervention), context)"""
    return scm.counterfactual({"target": target, "context": context, "do": intervention})
