# At the top
from prediction_module import TrinityPredictionEngine

class ThonocProofEngine:
    def __init__(self):
        self.translator = TranslationEngine()
        self.bridge = ConcreteTranslationBridge(PDNBridge())
        self.predictor = TrinityPredictionEngine()

        self.predict_keywords = {"predict", "prophecy", "forecast", "futures", "prophesy", "project", "divination", "foresight"}

    def is_predictive(self, query: str) -> bool:
        tokens = query.lower().split()
        return any(kw in tokens for kw in self.predict_keywords)

    def process(self, query: str, confirm_prediction: Optional[bool] = None) -> Dict[str, Any]:
        """Main processing pipeline. Includes smart prediction detection."""
        # 🔮 Detect if this is predictive in nature
        if self.is_predictive(query):
            if confirm_prediction is None:
                return {
                    "query": query,
                    "mode": "confirm_prediction",
                    "message": (
                        "This appears to be a forward-looking inquiry.\n\n"
                        "Would you like THŌNOC to run a metaphysical prediction based on this topic?"
                    )
                }

            if confirm_prediction:
                # Use keywords from query as input
                keywords = [word.strip(".,?") for word in query.lower().split()]
                prediction = self.predictor.predict(keywords)
                return {
                    "query": query,
                    "mode": "prediction",
                    "prediction": prediction,
                    "summary": self.sage_explanation(
                        query,
                        prediction["trinity"],
                        prediction["modal_status"],
                        prediction["coherence"]
                    )
                }
            else:
                return {
                    "query": query,
                    "mode": "prediction_skipped",
                    "message": "Prediction mode skipped. Proceeding with standard reasoning."
                }

        # 🧠 Else: normal interpretive mode
        trinity, translation_result = self.translate_query(query)
        lambda_expr = self.build_lambda_expr(trinity, translation_result)
        modal_status, coherence = self.evaluate_modal_status(trinity)
        fractal_data = self.map_to_fractal(trinity)

        return {
            "query": query,
            "mode": "inference",
            "trinity": trinity,
            "lambda_expr": lambda_expr,
            "modal_status": modal_status,
            "coherence": coherence,
            "fractal": fractal_data,
            "summary": self.sage_explanation(query, trinity, modal_status, coherence)
        }
