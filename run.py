from gesture_commands import ALL_COMMANDS
from gesture_controller import GestureController

gesture_contoller = GestureController(
    commands=[c() for c in ALL_COMMANDS]
)
gesture_contoller.run_inference()