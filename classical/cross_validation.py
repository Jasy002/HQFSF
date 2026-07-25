"""
Cross Validation Module.
"""

from sklearn.model_selection import StratifiedKFold

from utils.logger import setup_logger

logger = setup_logger()


class CrossValidator:

    def __init__(
        self,
        folds=5,
        random_state=42
    ):

        self.cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state
        )

    def split(self, X, y):

        return self.cv.split(X, y)