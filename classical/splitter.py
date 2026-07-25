"""
Train-Test Split Module.
"""

from sklearn.model_selection import train_test_split

from utils.logger import setup_logger

logger = setup_logger()


class DataSplitter:

    def __init__(
        self,
        test_size=0.2,
        random_state=42
    ):

        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):

        logger.info("Splitting dataset...")

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )