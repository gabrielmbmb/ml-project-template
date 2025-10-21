"""Data visualization utilities."""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class Visualizer:
    """Handle data visualization and plotting."""

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_distribution(
        self,
        data: pd.Series,
        title: str = "Distribution Plot",
        save_name: Optional[str] = None
    ):
        """Plot distribution of a variable."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data, kde=True)
        plt.title(title)
        plt.xlabel(data.name)
        plt.ylabel('Frequency')
        
        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        save_name: Optional[str] = None
    ):
        """Plot correlation matrix heatmap."""
        plt.figure(figsize=(12, 10))
        correlation = df.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        
        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        top_n: int = 20,
        save_name: Optional[str] = None
    ):
        """Plot feature importance."""
        # Sort features by importance
        indices = np.argsort(importance_values)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importance_values[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        
        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        save_name: Optional[str] = None
    ):
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_learning_curves(
        self,
        train_scores: List[float],
        val_scores: List[float],
        metric_name: str = "Score",
        save_name: Optional[str] = None
    ):
        """Plot learning curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_scores, label='Training', marker='o')
        plt.plot(val_scores, label='Validation', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        
        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Example usage of visualization utilities."""
    logging.basicConfig(level=logging.INFO)
    
    visualizer = Visualizer()
    
    # Example: Create sample plots
    # df = pd.read_csv("data/processed/data.csv")
    # visualizer.plot_correlation_matrix(df, save_name="correlation_matrix.png")
    
    logger.info("Visualization examples complete")


if __name__ == "__main__":
    main()