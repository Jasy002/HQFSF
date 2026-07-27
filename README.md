# Hybrid Quantum Feature Selection Framework (HQFSF)

> A hybrid quantum-classical framework for feature selection using Variational Quantum Circuits (VQCs) and classical machine learning algorithms.

---

## Overview

The **Hybrid Quantum Feature Selection Framework (HQFSF)** combines quantum computing and classical machine learning to perform intelligent feature selection before model training. The framework uses Variational Quantum Circuits (VQCs) to estimate feature importance and then trains classical machine learning models using only the selected features.

This project was developed as part of an **M.Tech research project** focused on Quantum Machine Learning and Hybrid AI systems.

---

## Key Features

- Hybrid Quantum–Classical Pipeline
- Variational Quantum Circuits (VQC)
- Quantum Feature Selection
- Multiple Machine Learning Models
- Configurable YAML-based Architecture
- Automated Evaluation Pipeline
- Benchmarking Utilities
- Result Export (CSV, Excel, JSON)
- Visualization Utilities
- Docker Support
- Modular and Extensible Design

---

# Project Structure

```text
HQFSF/
│
├── classical/
├── quantum/
├── pipeline/
├── models/
├── utils/
├── configs/
├── docs/
├── tests/
├── datasets/
│
├── main.py
├── run.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── setup.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── README.md
└── LICENSE
```

---

# Technology Stack

- Python 3.11+
- Qiskit
- Qiskit Aer
- NumPy
- Pandas
- Scikit-learn
- SciPy
- XGBoost
- Matplotlib
- Plotly
- PyYAML
- Joblib

---

# Installation

```bash
git clone https://github.com/Jasy002/HQFSF.git

cd HQFSF

pip install -r requirements.txt
```

---

# Running the Framework

```bash
python run.py
```

or

```bash
python main.py
```

---

# Training

```bash
python docs/train.py
```

---

# Evaluation

```bash
python docs/evaluate.py
```

---

# Benchmarking

```bash
python docs/benchmark.py
```

---

# Export Results

```bash
python docs/export_results.py
```

---

# Workflow

1. Load Dataset
2. Validate Dataset
3. Data Preprocessing
4. Feature Scaling
5. Quantum Feature Encoding
6. Variational Quantum Circuit
7. Expectation Value Computation
8. Quantum Feature Ranking
9. Feature Selection
10. Classical Model Training
11. Model Evaluation
12. Result Export

---

# Supported Machine Learning Models

- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
- XGBoost

---

# Quantum Components

- Quantum Feature Encoder
- Variational Quantum Circuit (VQC)
- EfficientSU2 Ansatz
- Quantum Measurement
- Expectation Value Calculator
- Quantum Feature Selector

---

# Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Feature Reduction
- Training Time

---

# Configuration

All project settings are stored in the `configs/` directory:

- `config.yaml`
- `datasets.yaml`
- `logging.yaml`
- `model.yaml`
- `quantum.yaml`

---

# Docker

Build the image:

```bash
docker compose build
```

Run the project:

```bash
docker compose up
```

---

# Testing

```bash
pytest
```

---

# License

This project is licensed under the MIT License.

---

# Author

**Jasmine Sultana**

M.Tech in Computer Science (Artificial Intelligence)

Vidyasagar University

GitHub: https://github.com/Jasy002/HQFSF