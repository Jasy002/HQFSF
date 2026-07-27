"""
==========================================================
HQFSF Results Export Script
==========================================================

Author : Jasmine Sultana
Project: Hybrid Quantum Feature Selection Framework
Version: 1.0.0

Exports experiment results into multiple formats
(CSV, Excel, JSON) for analysis, reporting,
and publication.
==========================================================
"""

import json
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
EXPORT_DIR = Path("exports")


def export_results():
    """Export experiment results."""

    print("=" * 60)
    print("HQFSF RESULTS EXPORT")
    print("=" * 60)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_file = RESULTS_DIR / "evaluation_results.csv"

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Result file not found: {csv_file}"
        )

    df = pd.read_csv(csv_file)

    # --------------------------------------------------
    # Export CSV
    # --------------------------------------------------
    csv_export = EXPORT_DIR / "results.csv"

    df.to_csv(csv_export, index=False)

    print(f"✓ CSV exported: {csv_export}")

    # --------------------------------------------------
    # Export Excel
    # --------------------------------------------------
    excel_export = EXPORT_DIR / "results.xlsx"

    with pd.ExcelWriter(excel_export) as writer:
        df.to_excel(
            writer,
            sheet_name="HQFSF Results",
            index=False
        )

    print(f"✓ Excel exported: {excel_export}")

    # --------------------------------------------------
    # Export JSON
    # --------------------------------------------------
    json_export = EXPORT_DIR / "results.json"

    with open(json_export, "w", encoding="utf-8") as file:
        json.dump(
            df.to_dict(orient="records"),
            file,
            indent=4
        )

    print(f"✓ JSON exported: {json_export}")

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("\nSummary")

    print("-" * 60)

    print(df)

    print("\nExport completed successfully.")


if __name__ == "__main__":
    export_results()