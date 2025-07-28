from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
import json
import os
from datetime import datetime
from scipy import stats
import pickle
from pathlib import Path

@dataclass
class BayesianPrediction:
    prediction: float
    confidence: float
    variance: float
    timestamp: str
    metadata: Dict

@dataclass
class ModelState:
    priors: Dict[str, float]
    likelihoods: Dict[str, float]
    posterior_history: List[Dict[str, float]]
    variance_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]

class BayesianMLModel(BaseModule):
    def __init__(self, data_path: str = "data/bayesian_model_data.pkl"):
        super().__init__()
        self.data_path = Path(data_path)
        self.model_state = ModelState(
            priors={},
            likelihoods={},
            posterior_history=[],
            variance_metrics={},
            performance_metrics={}
        )
        self.learning_rate = 0.01
        self.min_confidence_threshold = 0.7
        
    def initialize(self) -> None:
        """Initialize the model and load existing data if available"""
        self._initialized = True
        self.state = ModuleState.ACTIVE
        self._ensure_data_directory()
        self._load_model_state()
        
    def _ensure_data_directory(self) -> None:
        """Create data directory if it doesn't exist"""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _load_model_state(self) -> None:
        """Load model state from disk if available"""
        if self.data_path.exists():
            try:
                with open(self.data_path, 'rb') as f:
                    loaded_state = pickle.load(f)
                    self.model_state = loaded_state
                logging.info("Successfully loaded model state")
            except Exception as e:
                logging.error(f"Error loading model state: {e}")
                self._initialize_new_state()
        else:
            self._initialize_new_state()
            
    def _initialize_new_state(self) -> None:
        """Initialize a new model state with default values"""
        self.model_state = ModelState(
            priors={'default': 0.5},
            likelihoods={},
            posterior_history=[],
            variance_metrics={'global_variance': 0.0},
            performance_metrics={'accuracy': 0.0, 'confidence': 0.0}
        )
        self._save_model_state()
        
    def _save_model_state(self) -> None:
        """Persist model state to disk"""
        try:
            with open(self.data_path, 'wb') as f:
                pickle.dump(self.model_state, f)
            logging.info("Successfully saved model state")
        except Exception as e:
            logging.error(f"Error saving model state: {e}")

    def update_belief(self, hypothesis: str, evidence: Dict[str, float]) -> BayesianPrediction:
        """Update beliefs based on new evidence and return prediction"""
        prior = self.model_state.priors.get(hypothesis, 0.5)
        likelihood = self._calculate_likelihood(hypothesis, evidence)
        
        # Calculate posterior using Bayes' theorem
        marginal = self._calculate_marginal(evidence)
        posterior = (prior * likelihood) / marginal if marginal != 0 else prior
        
        # Calculate prediction confidence and variance
        confidence = self._calculate_confidence(posterior, evidence)
        variance = self._calculate_prediction_variance(posterior, evidence)
        
        # Create prediction object
        prediction = BayesianPrediction(
            prediction=posterior,
            confidence=confidence,
            variance=variance,
            timestamp=datetime.now().isoformat(),
            metadata={'evidence': evidence, 'prior': prior}
        )
        
        # Update model state
        self._update_model_metrics(prediction)
        self._save_model_state()
        
        return prediction

    def _calculate_likelihood(self, hypothesis: str, evidence: Dict[str, float]) -> float:
        """Calculate likelihood using historical data and current evidence"""
        likelihoods = []
        for key, value in evidence.items():
            evidence_key = f"{hypothesis}|{key}"
            if evidence_key in self.model_state.likelihoods:
                stored_likelihood = self.model_state.likelihoods[evidence_key]
                evidence_likelihood = stats.norm.pdf(value, loc=stored_likelihood, scale=0.1)
                likelihoods.append(evidence_likelihood)
        
        return np.prod(likelihoods) if likelihoods else 0.5

    def _calculate_marginal(self, evidence: Dict[str, float]) -> float:
        """Calculate marginal probability"""
        return sum(self.update_belief(h, evidence).prediction * p 
                  for h, p in self.model_state.priors.items())

    def _calculate_confidence(self, posterior: float, evidence: Dict[str, float]) -> float:
        """Calculate confidence score for prediction"""
        evidence_strength = np.mean(list(evidence.values()))
        prior_confidence = np.mean(list(self.model_state.priors.values()))
        return (posterior * evidence_strength * prior_confidence) ** (1/3)

    def _calculate_prediction_variance(self, posterior: float, evidence: Dict[str, float]) -> float:
        """Calculate variance for prediction refinement"""
        predictions = [p['prediction'] for p in self.model_state.posterior_history[-10:]]
        if predictions:
            return np.var(predictions + [posterior])
        return 0.0

    def _update_model_metrics(self, prediction: BayesianPrediction) -> None:
        """Update model metrics based on new prediction"""
        # Update posterior history
        self.model_state.posterior_history.append({
            'prediction': prediction.prediction,
            'confidence': prediction.confidence,
            'variance': prediction.variance,
            'timestamp': prediction.timestamp
        })
        
        # Update variance metrics
        self.model_state.variance_metrics['global_variance'] = np.mean(
            [p['variance'] for p in self.model_state.posterior_history[-50:]]
        )
        
        # Update performance metrics
        self.model_state.performance_metrics.update({
            'mean_confidence': np.mean([p['confidence'] for p in self.model_state.posterior_history[-50:]]),
            'prediction_stability': 1 / (1 + self.model_state.variance_metrics['global_variance'])
        })

    def validate(self) -> ValidationResult:
        """Validate model state and performance"""
        if not self._initialized:
            return ValidationResult(False, ["Module not initialized"], {})
            
        errors = []
        metrics = {
            'global_variance': self.model_state.variance_metrics['global_variance'],
            'mean_confidence': self.model_state.performance_metrics.get('mean_confidence', 0),
            'prediction_stability': self.model_state.performance_metrics.get('prediction_stability', 0)
        }
        
        # Validate probability constraints
        if any(p < 0 or p > 1 for p in self.model_state.priors.values()):
            errors.append("Invalid prior probabilities detected")
            
        # Validate confidence thresholds
        if metrics['mean_confidence'] < self.min_confidence_threshold:
            errors.append(f"Mean confidence below threshold: {metrics['mean_confidence']:.2f}")
            
        return ValidationResult(len(errors) == 0, errors, metrics)