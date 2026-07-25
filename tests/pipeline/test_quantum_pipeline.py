"""
Unit Test for QuantumPipeline.
"""

import numpy as np

from pipeline.quantum_pipeline import QuantumPipeline


def test_quantum_pipeline():
    """
    Test the Quantum Pipeline.
    """

    print("\n" + "=" * 70)
    print("            TESTING QUANTUM PIPELINE")
    print("=" * 70)

    # Dummy feature matrix
    X = np.array([
        [0.10, 0.20, 0.30, 0.40],
        [0.50, 0.60, 0.70, 0.80],
        [0.90, 1.00, 1.10, 1.20],
        [1.30, 1.40, 1.50, 1.60],
        [1.70, 1.80, 1.90, 2.00],
    ])

    pipeline = QuantumPipeline(
        n_qubits=4,
        layers=2,
        encoding="ry",
        entanglement="linear",
        backend_type="aer_simulator",
        shots=1024,
        selection_strategy="top_k",
        top_k=2,
    )

    pipeline.summary()

    result = pipeline.run(X)

    print("\n========== Quantum Pipeline Results ==========\n")

    print("Importance Scores")
    print(result["importance_scores"])

    print("\nRanking")
    print(result["ranking"])

    print("\nSelected Features")
    print(result["selected_features"])

    print("\nCounts")
    print(result["counts"])

    print("\n==============================================\n")

    # ------------------------------------------------
    # Assertions
    # ------------------------------------------------

    assert result is not None

    assert "importance_scores" in result
    assert "ranking" in result
    assert "selected_features" in result
    assert "counts" in result

    assert len(result["importance_scores"]) == X.shape[0]
    assert len(result["ranking"]) == X.shape[0]
    assert len(result["counts"]) == X.shape[0]

    assert len(result["selected_features"]) > 0

    print("✓ Quantum Pipeline Test Passed")


if __name__ == "__main__":
    test_quantum_pipeline()