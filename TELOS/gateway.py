from logos.kernel.lock_utils import lock_required
from .telos_nexus import TelosNexus

@lock_required
def run(lock_ctx, input_data):
    """
    Gateway for TELOS: only runs if the TLM token is valid.
    Delegates to your TelosNexus subsystem engine.
    """
    nexus = TelosNexus()
    # Pass in the data plus validated token
    return nexus.run(input_data=input_data, tlm_token=lock_ctx.token)
