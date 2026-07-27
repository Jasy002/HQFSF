"""
Dataset Splitter

Provides utilities for splitting datasets into
training, validation, and testing sets.
"""

from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """
    Dataset splitting utility.
    """

    def __init__(
        self,
        test_size=0.2,
        validation_size=0.1,
        random_state=42,
        stratify=True,
    ):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.stratify = stratify

    def train_test(self, X, y):
        """
        Train/Test split.
        """

        stratify_labels = y if self.stratify else None

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_labels,
        )

    def train_validation_test(self, X, y):
        """
        Train/Validation/Test split.
        """

        stratify_labels = y if self.stratify else None

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=self.test_size + self.validation_size,
            random_state=self.random_state,
            stratify=stratify_labels,
        )

        validation_ratio = (
            self.validation_size /
            (self.test_size + self.validation_size)
        )

        stratify_labels = y_temp if self.stratify else None

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1 - validation_ratio,
            random_state=self.random_state,
            stratify=stratify_labels,
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )