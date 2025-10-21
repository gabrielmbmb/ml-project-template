"""Tests for data loading and preprocessing."""

import pytest
import pandas as pd
import numpy as np
from src.data.dataset import DataLoader


class TestDataLoader:
    """Test DataLoader class."""
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        loader = DataLoader("data/raw", "data/processed")
        
        # Create sample data with duplicates and missing values
        df = pd.DataFrame({
            'A': [1, 2, 2, 4, np.nan],
            'B': [5, 6, 6, 8, 9],
            'C': [10, 11, 11, 13, 14]
        })
        
        processed = loader.preprocess_data(df)
        
        # Check duplicates removed
        assert len(processed) < len(df) or len(df) == len(df.drop_duplicates())
        
        # Check no missing values
        assert processed.isnull().sum().sum() == 0
    
    def test_split_data(self):
        """Test data splitting."""
        loader = DataLoader("data/raw", "data/processed")
        
        df = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'target': [0, 1] * 50
        })
        
        X_train, X_test, y_train, y_test = loader.split_data(
            df, 'target', test_size=0.2
        )
        
        # Check sizes
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20