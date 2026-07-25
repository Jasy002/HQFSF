"""
Runtime evaluation.
"""

from __future__ import annotations

import time

from typing import Dict

from .metrics import Metrics


class RuntimeMetrics(Metrics):
    """
    Runtime measurements.
    """

    def __init__(self):

        super().__init__()

        self.start_time = None

        self.end_time = None

    def start(self):

        self.start_time = time.perf_counter()

    def stop(self):

        self.end_time = time.perf_counter()

    def evaluate(self) -> Dict:

        runtime = 0.0

        if (

            self.start_time is not None

            and

            self.end_time is not None

        ):

            runtime = (

                self.end_time

                - self.start_time

            )

        return {

            "runtime_seconds": runtime
        }