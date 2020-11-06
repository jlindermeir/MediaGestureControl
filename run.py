from gesture_controller.commands import ALL_COMMANDS
from gesture_controller.controller import GestureController

gesture_contoller = GestureController(
    commands=[c() for c in ALL_COMMANDS]
)
gesture_contoller.run_inference()
