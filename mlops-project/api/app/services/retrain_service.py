"""Retrain service for retraining the Iris classification model with new data."""

import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)

from app.utils import get_project_root, LABEL_NAMES

logger = logging.getLogger(__name__)


class RetrainService:
    """Service for retraining the model with new data."""
    
    def __init__(self):
        """Initialize the retrain service."""
        self.project_root = get_project_root()
        self.model_dir = self.project_root / "model"
        self.retrain_dir = self.project_root / "retrain" / "new_data"
        self.retrain_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_csv(self, csv_path: Path) -> Tuple[bool, str]:
        """
        Validate that a CSV file has the required columns and valid data.
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Check required columns
            required_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
            
            # Check for empty dataframe
            if df.empty:
                return False, "CSV file is empty"
            
            # Check data types (should be numeric)
            for col in required_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    return False, f"Column '{col}' must be numeric"
            
            # Check for NaN values
            if df[required_cols].isna().any().any():
                return False, "CSV contains NaN values. Please remove or fill them"
            
            # Check target values are valid (0, 1, or 2)
            invalid_targets = df[~df['target'].isin([0, 1, 2])]
            if not invalid_targets.empty:
                return False, f"Target values must be 0, 1, or 2. Found: {invalid_targets['target'].unique().tolist()}"
            
            # Check reasonable ranges for features
            feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            for col in feature_cols:
                if (df[col] < 0).any() or (df[col] > 20).any():
                    logger.warning(f"Feature '{col}' has values outside typical range [0, 20]")
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Error validating CSV: {str(e)}"
    
    def save_uploaded_csv(self, file_content: bytes, filename: str) -> Path:
        """
        Save uploaded CSV file to retrain/new_data with timestamp.
        
        Args:
            file_content: Raw file content
            filename: Original filename
        
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_part = Path(filename).stem
        extension = Path(filename).suffix
        new_filename = f"{name_part}_{timestamp}{extension}"
        save_path = self.retrain_dir / new_filename
        
        with open(save_path, 'wb') as f:
            f.write(file_content)
        
        logger.info(f"Saved uploaded CSV to {save_path}")
        return save_path
    
    def retrain_from_csvs(self, csv_paths: List[Path]) -> Dict:
        """
        Retrain the model by combining base Iris data with new CSV files.
        
        Args:
            csv_paths: List of paths to CSV files with new data
        
        Returns:
            Dictionary with evaluation metrics and training info
        """
        logger.info(f"Starting retrain with {len(csv_paths)} CSV file(s)")
        
        # Load base Iris dataset
        iris = load_iris(as_frame=True)
        base_df = iris.frame.copy()
        base_df['target'] = iris.target
        
        # Load and combine new data
        new_dfs = []
        for csv_path in csv_paths:
            is_valid, error_msg = self.validate_csv(csv_path)
            if not is_valid:
                raise ValueError(f"Invalid CSV {csv_path}: {error_msg}")
            
            df_new = pd.read_csv(csv_path)
            new_dfs.append(df_new)
        
        if new_dfs:
            new_data = pd.concat(new_dfs, ignore_index=True)
            combined_df = pd.concat([base_df, new_data], ignore_index=True)
            logger.info(f"Combined dataset: {len(base_df)} base + {len(new_data)} new = {len(combined_df)} total")
        else:
            combined_df = base_df
            logger.info("Using only base dataset (no new data provided)")
        
        # Prepare features and target
        X = combined_df.drop(columns=['target'])
        y = combined_df['target']
        
        # Train/test split (same as notebook: 80/20, stratified, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features (fit on training, transform both)
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Grid search for best hyperparameters (same as notebook)
        params = {'C': [0.01, 0.1, 1, 10, 100]}
        grid = GridSearchCV(
            LogisticRegression(max_iter=1500, random_state=42),
            params,
            cv=5,
            scoring='accuracy'
        )
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        
        logger.info(f"Best hyperparameters: {grid.best_params_}")
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test_scaled)
        
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
        recall = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
        
        # Classification report as string
        report = classification_report(
            y_test, y_pred, 
            target_names=LABEL_NAMES,
            output_dict=False
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Save model and scaler
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / "iris_model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"Model and scaler saved to {self.model_dir}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "best_params": grid.best_params_,
            "classification_report": report,
            "confusion_matrix": cm,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": len(combined_df)
        }


# Global singleton instance
_retrain_service: Optional[RetrainService] = None


def get_retrain_service() -> RetrainService:
    """Get or create the global retrain service instance."""
    global _retrain_service
    if _retrain_service is None:
        _retrain_service = RetrainService()
    return _retrain_service

