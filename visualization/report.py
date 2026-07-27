"""
==============================================================
HQFSF Report Generator

Hybrid Quantum Feature Selection Framework (HQFSF)

Generates comprehensive experiment reports including:

• Selected Features
• Feature Importance
• Evaluation Metrics
• Runtime Statistics
==============================================================
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class ReportGenerator:
    """
    Generate HQFSF experiment reports.
    """

    def generate(
        self,
        results: dict,
        output_file: str | Path = "results/report.txt",
    ) -> None:
        """
        Generate a text report from HQFSF pipeline results.

        Parameters
        ----------
        results : dict
            Dictionary returned by HQFSFPipeline.run().

        output_file : str | Path, optional
            Output report file.
        """

        output_file = Path(output_file)

        # Create results directory if it does not exist
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = []

        report.append("=" * 80)
        report.append("HYBRID QUANTUM FEATURE SELECTION FRAMEWORK (HQFSF)")
        report.append("=" * 80)
        report.append(
            f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}"
        )
        report.append("")

        # ----------------------------------------------------
        # Experiment
        # ----------------------------------------------------

        report.append("=" * 80)
        report.append("EXPERIMENT INFORMATION")
        report.append("=" * 80)

        report.append(
            f"Experiment Name : {results.get('experiment_name', 'HQFSF Experiment')}"
        )

        report.append(
            f"Dataset         : {results.get('dataset_name', 'N/A')}"
        )

        report.append(
            f"Model           : {results.get('model_name', 'N/A')}"
        )

        report.append("")

        # ----------------------------------------------------
        # Selected Features
        # ----------------------------------------------------

        report.append("=" * 80)
        report.append("SELECTED FEATURES")
        report.append("=" * 80)

        selected_features = results.get(
            "selected_features",
            [],
        )

        if selected_features:

            report.append(
                f"Total Selected Features : {len(selected_features)}"
            )

            report.append("")

            for feature in selected_features:
                report.append(f"• {feature}")

        else:

            report.append("No selected features available.")

        report.append("")

        # ----------------------------------------------------
        # Feature Importance
        # ----------------------------------------------------

        report.append("=" * 80)
        report.append("FEATURE IMPORTANCE")
        report.append("=" * 80)

        scores = results.get(
            "importance_scores",
            [],
        )

        if len(scores):

            for idx, score in enumerate(scores):

                report.append(
                    f"Feature {idx + 1:<3}: {score:.6f}"
                )

        else:

            report.append("Importance scores not available.")

        report.append("")

        # ----------------------------------------------------
        # Evaluation Metrics
        # ----------------------------------------------------

        report.append("=" * 80)
        report.append("EVALUATION METRICS")
        report.append("=" * 80)

        metrics = results.get(
            "evaluation",
            {},
        )

        if metrics:

            for metric, value in metrics.items():

                report.append(
                    f"{metric:<25}: {value}"
                )

        else:

            report.append("Evaluation metrics not available.")

        report.append("")

        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------

        report.append("=" * 80)
        report.append("EXECUTION STATISTICS")
        report.append("=" * 80)

        runtime = results.get(
            "runtime",
            None,
        )

        if runtime is not None:

            report.append(
                f"Execution Time : {runtime:.4f} seconds"
            )

        else:

            report.append(
                "Execution Time : N/A"
            )

        report.append("")

        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        output_file.write_text(
            "\n".join(report),
            encoding="utf-8",
        )

        print(f"\nReport saved to: {output_file}")

    @staticmethod
    def summary() -> None:
        """
        Display report information.
        """

        print("\n" + "=" * 60)
        print("HQFSF Report Generator")
        print("=" * 60)

        print("Generated Sections")
        print("------------------")
        print("✓ Experiment Information")
        print("✓ Selected Features")
        print("✓ Feature Importance")
        print("✓ Evaluation Metrics")
        print("✓ Runtime Statistics")
        print("✓ Text Report")

        print("=" * 60)