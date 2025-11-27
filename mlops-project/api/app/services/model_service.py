"""Model service for loading and making predictions with the Iris classification model."""

import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from app.utils import get_project_root, LABEL_MAPPING

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing model and scaler loading and predictions."""
    
    def __init__(self):
        """Initialize the model service."""
        self.model: Optional[BaseEstimator] = None
        self.scaler: Optional[StandardScaler] = None
        self.model_path: Path = get_project_root() / "model" / "iris_model.pkl"
        self.scaler_path: Path = get_project_root() / "model" / "scaler.pkl"
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model and scaler from disk."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Model and scaler loaded successfully from {self.model_path.parent}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def reload_model(self) -> None:
        """Reload model and scaler from disk (useful after retraining)."""
        logger.info("Reloading model and scaler...")
        self._load_model()
    
    def predict_single(
        self, 
        sepal_length: float, 
        sepal_width: float, 
        petal_length: float, 
        petal_width: float
    ) -> Dict[str, any]:
        """
        Make a prediction for a single sample.
        
        Args:
            sepal_length: Sepal length in cm
            sepal_width: Sepal width in cm
            petal_length: Petal length in cm
            petal_width: Petal width in cm
        
        Returns:
            Dictionary with label, label_id, and probability
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not loaded")
        
        # Prepare input
        X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        label_id = int(self.model.predict(X_scaled)[0])
        probabilities = self.model.predict_proba(X_scaled)[0]
        probability = float(probabilities[label_id])
        label = LABEL_MAPPING[label_id]
        
        return {
            "label": label,
            "label_id": label_id,
            "probability": probability
        }
    
    def predict_batch(self, csv_path: Path) -> np.ndarray:
        """
        Make predictions for a batch of samples from a CSV file.
        
        Args:
            csv_path: Path to CSV file with feature columns
        
        Returns:
            Array of predictions
        """
        import pandas as pd
        
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not loaded")
        
        df = pd.read_csv(csv_path)
        required_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        X = df[required_cols].values
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions


# Global singleton instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get or create the global model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service

