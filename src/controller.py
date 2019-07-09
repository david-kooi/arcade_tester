import os
import zmq
import pickle
import subprocess
from time import sleep
from pynput.keyboard import Key, Controller

from zmq.eventloop.ioloop import IOLoop, PeriodicCallback 
from zmq.eventloop.zmqstream import ZMQStream

from zmq.eventloop.ioloop import IOLoop, PeriodicCallback 
from zmq.eventloop.zmqstream import ZMQStream

from logging import DEBUG, INFO, ERROR 
from utils import logging_util
from utils.logging_util import GetLogger


MAME_PATH = "/home/howardtang/Documents/mame"
LOG_PATH  = "/home/howardtang/Documents/arcade_tester/logs"





class Controller(object):
    def __init__(self, data_input_port, console_lvl=INFO, file_lvl=INFO, tp_on=False,\
                                                                      timeout=None):
        """
        @params data_input_port: tcp port on which to receive state data 
        @params console_lvl: global logging level for console output
        @params file_lvl: global logging level for file output
        @params tp_on: True to log throughput at TestPoints. False to disable
        @params timeout: Quit server after timeout. For unittesting purposes
        """
        self.__context = zmq.Context()

        # Setup global logging settings
        logging_util.SetGlobalLoggingLevel(consoleLevel=console_lvl, fileLevel=file_lvl,\
                                                                     globalLevel=True)
        # Create logger
        #log_path = SERVER_CONFIG.get("filepaths", "supervisor_log_filepath") 
        self.__logger = GetLogger("controller", LOG_PATH, logLevel=DEBUG,\
                                               fileLevel=DEBUG)
        self.__logger.debug("Logger Active")
        self.__logger.debug("PID: {}".format(os.getpid()))
        self.__logger.debug("Data Input Port: {}".format(data_input_port))


        # Setup state data socket 
        self.__data_input_socket = self.__context.socket(zmq.SUB)
        try:
            self.__data_input_socket.setsockopt(zmq.SUBSCRIBE,b"")
            self.__data_input_socket.connect("tcp://localhost:{}".format(data_input_port))
        except zmq.ZMQError as e:
            if e.errno == zmq.EADDRINUSE:
                self.__logger.error("Unable to connect data socket to port {}"\
                             .format(command_port))
                raise e


    def GetContext(self):
        """
        Return zmq context.
        """
        return self.__context
    
    def Start(self):
        """
        Start main event loop of the zmq kernel
        """
        try:
            self.__logger.debug("Controller Starting") 
            while True:
                self.__logger.debug("Received game state data")
                #game_state = self.__data_input_socket.recv()
                dict2 = pickle.load(open(self.__data_input_socket.recv(), "rb"))
                print dict2
                
        except KeyboardInterrupt:
            pass # Fall through to quit

        self.Quit()  
          
    def Quit(self):
        """
        Shut down server
        """
        self.__logger.info("Initiating server shutdown")

        # Must close all sockets before context will terminate
        self.__data_input_socket.close()
        self.__context.term() 


if __name__ == "__main__":

    program = MAME_PATH + "mame64 pacman"
    args    = "-window"

    controller = Controller(1111, DEBUG)
    

    controller.Start()


#    proc_mame = subprocess.Popen([program, args],stdin=subprocess.PIPE)
#    keyboard = Controller()
#    sleep(3)
#    keyboard.press(Key.esc)
#    


    
