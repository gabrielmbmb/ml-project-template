"""Dataset loading and preprocessing utilities."""

import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataLoader:
    """Handle data loading and basic preprocessing."""

    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self, filename: str) -> pd.DataFrame:
        """Load raw data from file."""
        filepath = self.raw_data_path / filename
        logger.info(f"Loading data from {filepath}")
        
        if filepath.suffix == '.csv':
            return pd.read_csv(filepath)
        elif filepath.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(filepath)
        elif filepath.suffix == '.parquet':
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing steps."""
        logger.info("Preprocessing data")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values (customize based on your needs)
        df = df.fillna(df.mean(numeric_only=True))
        
        return df

    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        logger.info(f"Splitting data with test_size={test_size}")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data."""
        filepath = self.processed_data_path / filename
        logger.info(f"Saving processed data to {filepath}")
        df.to_csv(filepath, index=False)


def main():
    """Main data processing pipeline."""
    # Example usage
    loader = DataLoader(
        raw_data_path="data/raw",
        processed_data_path="data/processed"
    )
    
    # Load, preprocess, and save data
    # df = loader.load_raw_data("your_data.csv")
    # df = loader.preprocess_data(df)
    # loader.save_processed_data(df, "processed_data.csv")
    
    logger.info("Data processing complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()