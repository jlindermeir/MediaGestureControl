from gesture_commands import UnpauseCommand
from gesture_controller import GestureController
from gesture_commands import MuteCommand

gesture_contoller = GestureController(
    commands=[
        UnpauseCommand(),
        MuteCommand()
    ]
)
gesture_contoller.run_inference()