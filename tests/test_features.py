"""Tests for feature engineering."""

import pytest
import pandas as pd
import numpy as np
from src.features.build_features import FeatureEngineer


class TestFeatureEngineer:
    """Test FeatureEngineer class."""
    
    def test_scale_features(self):
        """Test feature scaling."""
        engineer = FeatureEngineer()
        
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        
        scaled = engineer.scale_features(df, ['A', 'B'], method='standard')
        
        # Check mean is close to 0 and std is close to 1
        assert np.abs(scaled['A'].mean()) < 1e-10
        assert np.abs(scaled['A'].std() - 1.0) < 1e-10
    
    def test_encode_categorical(self):
        """Test categorical encoding."""
        engineer = FeatureEngineer()
        
        df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'value': [1, 2, 3, 4, 5]
        })
        
        encoded = engineer.encode_categorical(df, ['category'], method='onehot')
        
        # Check one-hot encoding created new columns
        assert 'category_B' in encoded.columns or 'category_C' in encoded.columns