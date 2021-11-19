# Hybrid Pac Man Simulator

* To add a new controller, make a subclass of `PacManController` located in `controller.py`.
* See `potential_field_controller.py` for an example. 

## Running the Simulator
* `python main.py`

## Game States
* A "state estimator" extracts the state of the game and makes it available to the controller. `game_state` is a dictionary that can be accessed with the following keys:
* TODO: Add available keys

## Controller Implementation
* Override the following methods to initialize, run, and tear down the controller:

```
    def initialize(self):
        """
        Runs before the game starts.
        """
        pass
    
    def compute_control(self, game_state):
        """
        Runs every time a game_state is available.
        """
        pass
    
    def tear_down(self):
        """
        Runs when the game ends.
        """
```

## Adding a  controller to the game
* Create a controller and pass it to the `arcade_controller` in `main.py
```
    ## Add controllers here
    pacman_controller = PotentialFieldController(keyboard)
    
    arcade_controller = ArcadeController(pacman_controller, keyboard, 1111, DEBUG) 
````
