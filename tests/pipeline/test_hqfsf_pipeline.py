"""
Integration Test for HQFSF Pipeline.
"""

from pipeline.hqfsf_pipeline import HQFSFPipeline


def test_hqfsf_pipeline():

    print("\n" + "=" * 80)
    print("              TESTING HQFSF PIPELINE")
    print("=" * 80)

    pipeline = HQFSFPipeline(
        dataset_path="datasets/sample.csv",
        target_column="target",
        n_qubits=4,
        layers=2,
        encoding="ry",
        entanglement="linear",
        backend_type="aer_simulator",
        shots=1024,
        scaler="standard",
        selection_strategy="top_k",
        top_k=3,
    )

    pipeline.summary()

    result = pipeline.run()

    print("\n========== HQFSF Results ==========\n")

    print("Selected Features")
    print(result["selected_features"])

    print("\nImportance Scores")
    print(result["importance_scores"])

    print("\nRanking")
    print(result["ranking"])

    print("\nEvaluation")
    for key, value in result["evaluation"].items():
        print(f"{key}:")
        print(value)
        print()

    assert "classical" in result
    assert "quantum" in result
    assert "evaluation" in result

    print("✓ HQFSF Pipeline Test Passed")


if __name__ == "__main__":
    test_hqfsf_pipeline()