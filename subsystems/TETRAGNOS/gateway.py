# gateway.py  (place inside each subsystem folder)

from subsystems.TETRAGNOS.lock_utils import lock_required

@lock_required
def run(token="INVALID", *args, **kwargs):
    """
    Single entry-point for the subsystem.
    Today: just a placeholder.
    Later: call the real work function.
    """
    print("✅ Subsystem gateway reached — replace this with real call")
    return True    # placeholder result
