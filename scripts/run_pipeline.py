"""
Run Complete HQFSF Pipeline.
"""

from pipeline.hqfsf_pipeline import HQFSFPipeline

from visualization.report import ReportGenerator

from utils.logger import get_logger

logger = get_logger(__name__)


def main():

    logger.info("=" * 70)
    logger.info("RUNNING COMPLETE HQFSF PIPELINE")
    logger.info("=" * 70)

    pipeline = HQFSFPipeline()

    results = pipeline.run()

    report = ReportGenerator()

    report.generate(results)

    print("\nPipeline Finished Successfully")

    print("\nReport Generated")


if __name__ == "__main__":
    main()