"""Model prediction utilities."""

import logging
import joblib
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Predictor:
    """Handle model predictions."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()

    def load_model(self):
        """Load trained model."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)

    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Make predictions on input data."""
        logger.info("Making predictions...")
        predictions = self.model.predict(data)
        return predictions

    def predict_proba(
        self,
        data: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict probabilities for input data."""
        logger.info("Predicting probabilities...")
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(data)
            return probabilities
        else:
            raise AttributeError("Model does not support probability predictions")

    def predict_batch(
        self,
        data_path: str,
        output_path: str = None,
        batch_size: int = 1000
    ):
        """Make predictions on large datasets in batches."""
        logger.info(f"Processing data from {data_path} in batches")
        
        # Read data in chunks
        chunks = pd.read_csv(data_path, chunksize=batch_size)
        predictions = []
        
        for chunk in chunks:
            batch_preds = self.predict(chunk)
            predictions.extend(batch_preds)
        
        predictions = np.array(predictions)
        
        if output_path:
            logger.info(f"Saving predictions to {output_path}")
            pd.DataFrame({'predictions': predictions}).to_csv(output_path, index=False)
        
        return predictions


def main():
    """Main prediction pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to input data')
    parser.add_argument('--output', type=str, help='Path to save predictions')
    
    args = parser.parse_args()
    
    predictor = Predictor(args.model)
    data = pd.read_csv(args.data)
    predictions = predictor.predict(data)
    
    if args.output:
        pd.DataFrame({'predictions': predictions}).to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")
    else:
        print(predictions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()