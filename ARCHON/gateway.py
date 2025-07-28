# logos_system/subsystems/ARCHON/gateway.py

from typing import Dict, Any
from logos.kernel.lock_utils import lock_required
from .archon_nexus import ArchonNexus

@lock_required
def run(lock_ctx, output_data: Dict[str, Any]):
    """
    Gateway for ARCHON: only runs if the TLM token is valid.
    Delegates to ArchonNexus for semantic enrichment and final formatting.
    """
    nexus = ArchonNexus()
    return nexus.run(output_data=output_data, tlm_token=lock_ctx.token)
