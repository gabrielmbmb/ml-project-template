"""Model training utilities."""

import logging
import joblib
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handle model training and evaluation."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.metrics = {}

    def initialize_model(self):
        """Initialize model based on configuration."""
        model_type = self.config['model']['type']
        params = self.config['model']['hyperparameters']
        
        logger.info(f"Initializing {model_type} model")
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(**params)
        # Add more model types as needed
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ):
        """Train the model."""
        logger.info("Training model...")
        
        if self.model is None:
            self.initialize_model()
        
        self.model.fit(X_train, y_train)
        logger.info("Training complete")

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        logger.info("Evaluating model...")
        
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        logger.info(f"Evaluation metrics: {self.metrics}")
        return self.metrics

    def save_model(self, filepath: str = None):
        """Save trained model."""
        if filepath is None:
            save_dir = Path(self.config['model']['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_dir / self.config['model']['model_name']
        
        logger.info(f"Saving model to {filepath}")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str):
        """Load a trained model."""
        logger.info(f"Loading model from {filepath}")
        self.model = joblib.load(filepath)


def main():
    """Main training pipeline."""
    logging.basicConfig(level=logging.INFO)
    
    trainer = ModelTrainer()
    
    # Load your processed data
    # X_train, X_test, y_train, y_test = load_data()
    
    # Train and evaluate
    # trainer.train(X_train, y_train)
    # metrics = trainer.evaluate(X_test, y_test)
    # trainer.save_model()
    
    logger.info("Training pipeline complete")


if __name__ == "__main__":
    main()