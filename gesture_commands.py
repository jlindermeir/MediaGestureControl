import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from playerdo.main import find_players, do_command


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

        score = np.sum([
            t[1] for t in prediction_tuples
            if t[0] in self.valid_labels
        ])

        print(f'{str(self)}: {score}')

        if score >= self.threshold:
            self.last_triggered = now
            print(f'{str(self)} triggered!')
            self.action()

    def action(self):
        raise NotImplementedError


class PlayerDoCommand(BaseCommand):
    def __init__(self, command, **kwargs):
        super().__init__(**kwargs)
        self.command = command

    def action(self):
        do_command(self.command, find_players())

    def __str__(self):
        return f'"{self.command}" command'


class UnpauseCommand(PlayerDoCommand):
    def __init__(self):
        super().__init__(
            command='unpause',
            valid_labels=['Nodding', 'Thumb up'],
            threshold=0.8,
            cooldown=2
        )


class MuteCommand(PlayerDoCommand):
    def __init__(self):
        super().__init__(
            command='pause',
            valid_labels=["Putting finger to mouth"],
            threshold=0.8,
            cooldown=2
        )