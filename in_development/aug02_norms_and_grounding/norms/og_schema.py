from core.logos_validator_hub import validator_gate

class OGSchema:
    """
    Defines the basic utility structure for 'objective good' across actions.
    """
    def __init__(self):
        # Example: weights for categories of actions
        self.weights = {
            'help': 1.0,
            'inform': 0.8,
            'protect': 0.9,
            'avoid_harm': 1.0
        }

    @validator_gate
    def evaluate(self, action: str, context: dict):
        """Return a utility score for a given action under context."""
        base = self.weights.get(action, 0.5)
        # Contextual modifiers
        modifier = context.get('importance', 1.0)
        return base * modifier
