import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from playerdo.main import find_players, do_command


class BaseCommand:
    def __init__(self, valid_labels: List[str], threshold: float, cooldown: Optional[float] = None):
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

        if score >= self.threshold:
            self.last_triggered = now
            print(f'{str(self)} triggered!')
            self.action()

    def action(self):
        raise NotImplementedError


class PlayerDoCommand(BaseCommand):
    def __init__(self, command: str, threshold: float = 0.8, cooldown: Optional[float] = 2, **kwargs):
        super().__init__(threshold=threshold, cooldown=cooldown, **kwargs)
        self.command = command

    def action(self):
        do_command(self.command, find_players())

    def __str__(self):
        return f'"{self.command}" command'


class UnpauseCommand(PlayerDoCommand):
    def __init__(self):
        super().__init__(
            command='unpause',
            valid_labels=['Nodding', 'Thumb up']
        )


class MuteCommand(PlayerDoCommand):
    def __init__(self):
        super().__init__(
            command='pause',
            valid_labels=["Putting finger to mouth"]
        )


class NextCommand(PlayerDoCommand):
    def __init__(self):
        super().__init__(
            command='next',
            valid_labels=["Pointing left", "Swiping left"],
            threshold=0.7,
            cooldown=1
        )


class PrevCommand(PlayerDoCommand):
    def __init__(self):
        super().__init__(
            command='prev',
            valid_labels=["Pointing right", "Swiping right"],
            threshold=0.7,
            cooldown=1
        )


ALL_COMMANDS = [UnpauseCommand, MuteCommand, PrevCommand, NextCommand]
