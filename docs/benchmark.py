"""
==========================================================
HQFSF Benchmark Script
==========================================================

Author : Jasmine Sultana
Project: Hybrid Quantum Feature Selection Framework
Version: 1.0.0

This script benchmarks all supported machine learning
models using the features selected by HQFSF.
==========================================================
"""

import time
import pandas as pd

from models.model_factory import ModelFactory
from pipeline.classical_pipeline import ClassicalPipeline
from pipeline.quantum_pipeline import QuantumPipeline
from pipeline.evaluation_pipeline import EvaluationPipeline


SUPPORTED_MODELS = [
    "random_forest",
    "svm",
    "logistic_regression",
    "xgboost"
]


def benchmark():
    """Run benchmark on all supported models."""

    print("=" * 60)
    print("HQFSF BENCHMARK")
    print("=" * 60)

    # -------------------------
    # Classical preprocessing
    # -------------------------
    classical = ClassicalPipeline()

    data = classical.run()

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # -------------------------
    # Quantum Feature Selection
    # -------------------------
    quantum = QuantumPipeline()

    q_results = quantum.run(X_train, y_train)

    selected = q_results["selected_features"]

    X_train = X_train[:, selected]
    X_test = X_test[:, selected]

    evaluator = EvaluationPipeline()

    results = []

    for model_name in SUPPORTED_MODELS:

        print(f"\nBenchmarking {model_name}")

        model = ModelFactory.create(model_name)

        start = time.perf_counter()

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        elapsed = time.perf_counter() - start

        metrics = evaluator.evaluate(
            y_true=y_test,
            y_pred=predictions,
            selected_features=selected,
            total_features=X_train.shape[1]
        )

        metrics["model"] = model_name
        metrics["training_time"] = round(elapsed, 4)

        results.append(metrics)

    df = pd.DataFrame(results)

    print("\nBenchmark Summary")
    print(df)

    df.to_csv(
        "results/benchmark_results.csv",
        index=False
    )

    print("\nResults saved to results/benchmark_results.csv")


if __name__ == "__main__":
    benchmark()