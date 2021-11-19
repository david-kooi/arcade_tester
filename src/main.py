import subprocess
from logging import DEBUG, INFO
from pynput.keyboard import Key, Controller
from arcade_controller import ArcadeController
from potential_field_controller import PotentialFieldController


if __name__ == "__main__":

    # Start the state estimator
    args = ["python"]
    args.append("state_estimator.py")
    state_estimator_process = subprocess.Popen(args,stdin=subprocess.PIPE)


    # Get keyboard emulator
    keyboard = Controller()

    ## Add controllers here
    pacman_controller = PotentialFieldController(keyboard)
    
    arcade_controller = ArcadeController(pacman_controller, keyboard, 1111, DEBUG) 

    # Blocks until CTRL-C interrupt
    arcade_controller.start()

    state_estimator_process.kill()
