"""Tests for model training and prediction."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.models.train import ModelTrainer


class TestModelTrainer:
    """Test ModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        X_train = pd.DataFrame(X[:80])
        X_test = pd.DataFrame(X[80:])
        y_train = pd.Series(y[:80])
        y_test = pd.Series(y[80:])
        
        return X_train, X_test, y_train, y_test
    
    def test_model_initialization(self):
        """Test model initialization."""
        trainer = ModelTrainer()
        trainer.initialize_model()
        
        assert trainer.model is not None
    
    def test_model_training(self, sample_data):
        """Test model training."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer()
        trainer.train(X_train, y_train)
        
        assert trainer.model is not None
    
    def test_model_evaluation(self, sample_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer()
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1