"""
Main Entry Point for HQFSF.

Hybrid Quantum Feature Selection Framework
Using Variational Quantum Circuits
"""

from pipeline.hqfsf_pipeline import HQFSFPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """
    Execute the HQFSF pipeline.
    """

    logger.info("=" * 70)
    logger.info("Starting HQFSF Project")
    logger.info("=" * 70)

    pipeline = HQFSFPipeline(
        dataset_path="datasets/sample.csv",
        target_column="target",

        # Quantum Configuration
        n_qubits=4,
        layers=2,
        encoding="ry",
        entanglement="linear",
        backend_type="aer_simulator",
        shots=1024,

        # Classical Configuration
        scaler="standard",
        test_size=0.20,
        random_state=42,

        # Feature Selection
        selection_strategy="top_k",
        top_k=5,
        threshold=0.5,
    )

    pipeline.summary()

    results = pipeline.run()

    print("\n" + "=" * 70)
    print("HQFSF EXECUTION COMPLETED")
    print("=" * 70)

    print("\nSelected Features:")
    print(results["selected_features"])

    print("\nFeature Importance Scores:")
    print(results["importance_scores"])

    print("\nFeature Ranking:")
    print(results["ranking"])

    print("\nEvaluation Metrics:")
    for metric, value in results["evaluation"].items():
        print(f"{metric}: {value}")

    logger.info("HQFSF execution completed successfully.")


if __name__ == "__main__":
    main()