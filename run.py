"""
HQFSF Project Entry Point
"""

from pipeline.hqfsf_pipeline import HQFSFPipeline


def main():
    print("=" * 70)
    print("Hybrid Quantum Feature Selection Framework")
    print("=" * 70)

    pipeline = HQFSFPipeline()

    results = pipeline.run()

    print("\nExecution Completed Successfully.")

    return results


if __name__ == "__main__":
    main()