from logos.kernel.lock_utils import lock_required
from .tetranose_nexus import TetranoseNexus

@lock_required
def run(lock_ctx, input_data):
    """
    Gateway for TETRAGNOS: only runs if the TLM token is valid.
    Delegates to your TetranoseNexus translation engine.
    """
    nexus = TetranoseNexus()
    # Pass in the data plus validated token
    return nexus.run(input_data=input_data, tlm_token=lock_ctx.token)
