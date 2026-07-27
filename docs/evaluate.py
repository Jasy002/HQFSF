"""
==========================================================
HQFSF Evaluation Script
==========================================================

Author : Jasmine Sultana
Project: Hybrid Quantum Feature Selection Framework
Version: 1.0.0

Evaluates a trained machine learning model using
features selected by the HQFSF quantum framework.
==========================================================
"""

from pathlib import Path
import pandas as pd

from pipeline.classical_pipeline import ClassicalPipeline
from pipeline.quantum_pipeline import QuantumPipeline
from pipeline.evaluation_pipeline import EvaluationPipeline
from models.model_factory import ModelFactory


MODEL_NAME = "random_forest"


def evaluate():
    """Evaluate a trained model."""

    print("=" * 60)
    print("HQFSF MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # Data preprocessing
    # --------------------------------------------------
    classical = ClassicalPipeline()

    data = classical.run()

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # --------------------------------------------------
    # Quantum Feature Selection
    # --------------------------------------------------
    quantum = QuantumPipeline()

    q_result = quantum.run(X_train, y_train)

    selected_features = q_result["selected_features"]

    X_train = X_train[:, selected_features]
    X_test = X_test[:, selected_features]

    print(f"\nSelected Features : {len(selected_features)}")

    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------
    model = ModelFactory.create(MODEL_NAME)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    evaluator = EvaluationPipeline()

    metrics = evaluator.evaluate(
        y_true=y_test,
        y_pred=predictions,
        selected_features=selected_features,
        total_features=data["X_train"].shape[1]
    )

    print("\nEvaluation Results")
    print("-" * 60)

    for key, value in metrics.items():
        print(f"{key:25} : {value}")

    # --------------------------------------------------
    # Save Results
    # --------------------------------------------------
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame([metrics])

    output_file = output_dir / "evaluation_results.csv"

    result_df.to_csv(output_file, index=False)

    print("\nResults saved to:")
    print(output_file.resolve())


if __name__ == "__main__":
    evaluate()