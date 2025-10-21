# ML Project Template

A comprehensive Python machine learning project template following best practices for data science and ML development.

## 🎯 Project Overview

This template provides a standardized structure for machine learning projects, making it easier to:
- Organize code, data, and experiments
- Collaborate with team members
- Reproduce results
- Deploy models to production

## 📁 Project Structure

```
ml-project-template/
│
├── data/
│   ├── raw/                # Original, immutable data
│   ├── processed/          # Cleaned and transformed data
│   ├── interim/            # Intermediate data transformations
│   └── external/           # Data from third-party sources
│
├── notebooks/              # Jupyter notebooks for exploration
│   ├── exploratory/        # EDA and experimentation
│   └── reports/            # Final analysis and visualizations
│
├── src/                    # Source code for the project
│   ├── __init__.py
│   ├── data/               # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── features/           # Feature engineering
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/             # Model definitions and training
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── visualization/      # Plotting and visualization
│   │   ├── __init__.py
│   │   └── visualize.py
│   └── utils/              # Utility functions
│       ├── __init__.py
│       └── helpers.py
│
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
├── models/                 # Trained models and model artifacts
│   └── .gitkeep
│
├── configs/                # Configuration files
│   ├── config.yaml
│   └── logging_config.yaml
│
├── docs/                   # Documentation files
│   └── .gitkeep
│
├── scripts/                # Utility scripts
│   └── .gitkeep
│
├── reports/                # Generated analysis and reports
│   └── figures/            # Generated graphics
│       └── .gitkeep
│
├── .gitignore             # Git ignore file
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation script
├── pyproject.toml        # Modern Python project configuration
├── Makefile              # Automation commands
└── README.md             # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management
- Git for version control

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gabrielmbmb/ml-project-template.git
   cd ml-project-template
   ```

2. **Create a virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n ml-project python=3.9
   conda activate ml-project
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
   # For development (includes testing and formatting tools)
   pip install -r requirements-dev.txt
   ```

4. **Install the project as a package**
   ```bash
   pip install -e .
   ```

## 💻 Usage

### Data Preparation

1. Place your raw data in `data/raw/`
2. Run data preprocessing:
   ```bash
   python src/data/dataset.py
   ```

### Feature Engineering

```bash
python src/features/build_features.py
```

### Model Training

```bash
python src/models/train.py --config configs/config.yaml
```

### Making Predictions

```bash
python src/models/predict.py --model models/best_model.pkl --data data/processed/test.csv
```

### Running Notebooks

```bash
jupyter notebook notebooks/
```

## 🧪 Testing

Run tests using pytest:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_models.py
```

## 📊 Experiment Tracking

This template supports multiple experiment tracking tools:

- **MLflow**: For tracking experiments, parameters, and metrics
- **Weights & Biases**: For advanced experiment visualization
- **TensorBoard**: For deep learning experiments

Configure your preferred tool in `configs/config.yaml`.

## 🔧 Development

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/
pylint src/

# Type checking
mypy src/
```

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

## 📈 Model Deployment

### Local Deployment

```bash
python src/models/serve.py
```

### Docker Deployment

```bash
docker build -t ml-model .
docker run -p 5000:5000 ml-model
```

## 📝 Configuration

Edit `configs/config.yaml` to customize:

- Data paths
- Model hyperparameters
- Training settings
- Logging configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [Python Packaging Guide](https://packaging.python.org/)
- ML/AI community best practices

## 📧 Contact

Gabriel Martín Blázquez - [@gabrielmbmb](https://github.com/gabrielmbmb)

Project Link: [https://github.com/gabrielmbmb/ml-project-template](https://github.com/gabrielmbmb/ml-project-template)

---

**Happy Machine Learning! 🚀🤖**