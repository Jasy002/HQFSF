"""
==========================================================
HQFSF Project Entry Point
==========================================================

Hybrid Quantum Feature Selection Framework
Using Variational Quantum Circuits

Author  : Jasmine Sultana
Version : 1.0.0
==========================================================
"""

import sys

from pipeline.hqfsf_pipeline import HQFSFPipeline


def main() -> int:
    """Execute the HQFSF pipeline."""

    print("=" * 70)
    print("Hybrid Quantum Feature Selection Framework")
    print("=" * 70)

    try:
        pipeline = HQFSFPipeline()

        results = pipeline.run()

        print("\nExecution Completed Successfully.")
        return 0

    except Exception as exc:
        print(f"\nExecution Failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())