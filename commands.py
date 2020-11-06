import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np


class BaseCommand:
    def __init__(self, valid_labels: List[str], threshold: float, cooldown: Optional[int] = None):
        self.valid_labels = valid_labels
        self.threshold = threshold
        self.cooldown = cooldown or 0

        self.last_triggered = time.perf_counter() - cooldown

    def __call__(self, predictions: Dict[str, List[Tuple[str, int]]]):
        now = time.perf_counter()
        if self.cooldown is not None and now - self.last_triggered < self.cooldown:
            return

        prediction_tuples = predictions['sorted_predictions']

        score = np.mean([
            t[1] for t in prediction_tuples
            if t[0] in self.valid_labels
        ])

        if score >= self.threshold:
            self.last_triggered = now
            self.action()

    def action(self):
        raise NotImplementedError
