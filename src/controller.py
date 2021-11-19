import os
from time import sleep
from logging import INFO, DEBUG
from pynput.keyboard import Key, Controller

from utils import logging_util
from utils.logging_util import GetLogger



AT_ROOT   = os.environ["AT_ROOT"]
LOG_PATH  = os.path.join(AT_ROOT, "logs")



class PacManController(object):
    def __init__(self, keyboard, console_lvl=INFO, file_lvl=DEBUG):

        """
        @params keyboard: A punput keyboard 
        @params console_lvl: global logging level for console output
        @params file_lvl: global logging level for file output
        """
        self.__keyboard = keyboard 

        # Setup global logging settings
        logging_util.SetGlobalLoggingLevel(consoleLevel=console_lvl, fileLevel=file_lvl,\
                                                                     globalLevel=True)
        # Create logger
        #log_path = SERVER_CONFIG.get("filepaths", "supervisor_log_filepath") 
        self.logger = GetLogger("pac_man_controller", LOG_PATH, logLevel=DEBUG,\
                                               fileLevel=DEBUG)
        self.logger.debug("Logger Active") 
        

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
        pass


    ## Methods to control MAME
    #
    def request_screen(self):
        self.press_key(Key.f12)

    # Command PacMan to walk in a certain direction
    # These commands are latching. 
    # I.e, PacMan will continue in the direction of the
    # Last sent command. 
    def send_right(self):
        self.press_key(Key.right) 

    def send_left(self):
        self.press_key(Key.left)

    def send_down(self):
        self.press_key(Key.down)

    def send_up(self):
        self.press_key(Key.up)

    def press_key(self, key):
        self.__keyboard.press(key)
        sleep(0.01)
        self.__keyboard.release(key)
