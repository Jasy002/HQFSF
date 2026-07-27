"""
==========================================================
HQFSF Training Script
==========================================================

Author : Jasmine Sultana
Project : Hybrid Quantum Feature Selection Framework
Version : 1.0.0

This script performs the complete HQFSF workflow:

1. Load Dataset
2. Data Preprocessing
3. Quantum Feature Selection
4. Train ML Model
5. Evaluate Performance
6. Save Model
7. Save Results

==========================================================
"""

from pathlib import Path
import joblib
import pandas as pd

from pipeline.classical_pipeline import ClassicalPipeline
from pipeline.quantum_pipeline import QuantumPipeline
from pipeline.evaluation_pipeline import EvaluationPipeline
from models.model_factory import ModelFactory


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_NAME = "random_forest"

MODEL_DIR = Path("saved_models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def train():
    """Run the complete HQFSF training pipeline."""

    print("=" * 60)
    print("HQFSF TRAINING")
    print("=" * 60)

    # ------------------------------------------------------
    # Classical Pipeline
    # ------------------------------------------------------

    print("\nLoading dataset...")

    classical = ClassicalPipeline()

    data = classical.run()

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print("Dataset loaded successfully.")

    # ------------------------------------------------------
    # Quantum Feature Selection
    # ------------------------------------------------------

    print("\nRunning Quantum Feature Selection...")

    quantum = QuantumPipeline()

    quantum_results = quantum.run(
        X_train,
        y_train
    )

    selected_features = quantum_results["selected_features"]

    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    print(f"Selected Features : {len(selected_features)}")

    # ------------------------------------------------------
    # Train Model
    # ------------------------------------------------------

    print(f"\nTraining {MODEL_NAME}...")

    model = ModelFactory.create(MODEL_NAME)

    model.fit(
        X_train_selected,
        y_train
    )

    print("Training completed.")

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    predictions = model.predict(
        X_test_selected
    )

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------

    evaluator = EvaluationPipeline()

    metrics = evaluator.evaluate(
        y_true=y_test,
        y_pred=predictions,
        selected_features=selected_features,
        total_features=X_train.shape[1]
    )

    # ------------------------------------------------------
    # Save Model
    # ------------------------------------------------------

    model_path = MODEL_DIR / f"{MODEL_NAME}.joblib"

    joblib.dump(
        model,
        model_path
    )

    # ------------------------------------------------------
    # Save Metrics
    # ------------------------------------------------------

    metrics_df = pd.DataFrame([metrics])

    metrics_path = RESULT_DIR / "training_results.csv"

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print("\nTraining Summary")
    print("-" * 60)

    print(f"Model              : {MODEL_NAME}")
    print(f"Original Features  : {X_train.shape[1]}")
    print(f"Selected Features  : {len(selected_features)}")

    for metric, value in metrics.items():
        print(f"{metric:20}: {value}")

    print("\nSaved Files")
    print("-" * 60)
    print(f"Model   : {model_path}")
    print(f"Results : {metrics_path}")

    print("\nHQFSF Training Completed Successfully.")


if __name__ == "__main__":
    train()