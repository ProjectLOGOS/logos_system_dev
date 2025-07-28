from causal.scm import SCM

def apply_intervention(scm: SCM, var, value):
    """Convenience wrapper around scm.do"""
    return scm.do({var: value})
