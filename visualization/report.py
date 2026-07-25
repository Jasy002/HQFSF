"""
Report Generator.

Generates experiment reports for HQFSF.
"""

from __future__ import annotations

from datetime import datetime


class ReportGenerator:
    """
    Generate experiment reports.
    """

    def __init__(self):
        pass

    def generate(
        self,
        experiment_name: str,
        selected_features,
        importance_scores,
        evaluation_metrics: dict,
        runtime: float,
        output_file: str = "results/report.txt",
    ) -> None:
        """
        Generate a text report.
        """

        report = []

        report.append("=" * 80)
        report.append("HYBRID QUANTUM FEATURE SELECTION FRAMEWORK (HQFSF)")
        report.append("=" * 80)

        report.append(f"Experiment : {experiment_name}")
        report.append(f"Generated  : {datetime.now()}")
        report.append("")

        report.append("=" * 80)
        report.append("SELECTED FEATURES")
        report.append("=" * 80)

        report.append(str(selected_features))
        report.append("")

        report.append("=" * 80)
        report.append("FEATURE IMPORTANCE")
        report.append("=" * 80)

        for index, score in enumerate(importance_scores):
            report.append(f"Feature {index + 1:<3} : {score:.6f}")

        report.append("")

        report.append("=" * 80)
        report.append("EVALUATION METRICS")
        report.append("=" * 80)

        for metric, value in evaluation_metrics.items():
            report.append(f"{metric:<25}: {value}")

        report.append("")

        report.append("=" * 80)
        report.append("EXECUTION TIME")
        report.append("=" * 80)

        report.append(f"{runtime:.4f} seconds")

        report.append("")
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        with open(output_file, "w", encoding="utf-8") as file:
            file.write("\n".join(report))

        print(f"\nReport saved to: {output_file}")

    def summary(self):

        print("\n" + "=" * 60)
        print(" Report Generator ")
        print("=" * 60)

        print("Generates")
        print("✓ Experiment Summary")
        print("✓ Selected Features")
        print("✓ Feature Importance")
        print("✓ Evaluation Metrics")
        print("✓ Runtime")
        print("✓ Output Report")

        print("=" * 60)