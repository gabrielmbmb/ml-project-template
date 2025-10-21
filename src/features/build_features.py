"""Feature engineering and transformation utilities."""

import logging
from typing import List, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handle feature engineering and transformations."""

    def __init__(self):
        self.scaler = None
        self.encoder = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        logger.info("Creating new features")
        
        # Example feature engineering
        # Add your custom feature engineering logic here
        
        return df

    def scale_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'standard'
    ) -> pd.DataFrame:
        """Scale numerical features."""
        logger.info(f"Scaling features using {method} scaler")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """Encode categorical features."""
        logger.info(f"Encoding categorical features using {method}")
        
        if method == 'onehot':
            df = pd.get_dummies(df, columns=columns, drop_first=True)
        elif method == 'label':
            for col in columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        return df

    def select_features(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        k: int = 10
    ) -> pd.DataFrame:
        """Select top k features based on importance."""
        from sklearn.feature_selection import SelectKBest, f_classif
        
        logger.info(f"Selecting top {k} features")
        
        selector = SelectKBest(f_classif, k=k)
        selected_features = selector.fit_transform(df, target)
        
        selected_columns = df.columns[selector.get_support()]
        return pd.DataFrame(selected_features, columns=selected_columns)


def main():
    """Main feature engineering pipeline."""
    logger.info("Feature engineering pipeline started")
    
    # Example usage
    # engineer = FeatureEngineer()
    # df = pd.read_csv("data/processed/processed_data.csv")
    # df = engineer.create_features(df)
    # df = engineer.scale_features(df, numerical_columns)
    # df = engineer.encode_categorical(df, categorical_columns)
    
    logger.info("Feature engineering complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()