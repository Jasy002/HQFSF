"""
Unit Test for Logger.
"""

from utils.logger import get_logger


def test_logger():

    logger = get_logger("HQFSF-Test")

    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    assert logger is not None

    print("✓ Logger Test Passed")


if __name__ == "__main__":
    test_logger()