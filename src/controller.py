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



# Path to arcade tester project root
AT_ROOT   = os.environ["AT_ROOT"]
LOG_PATH  = os.path.join(AT_ROOT, "logs")

# Path to mame file system
MAME_PATH = os.environ["MAME_PATH"]



class ArcadeController(object):
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
            self.__data_input_socket.setsockopt(zmq.RCVTIMEO, 1000)
            self.__data_input_socket.connect("tcp://localhost:{}".format(data_input_port))
        except zmq.ZMQError as e:
            if e.errno == zmq.EADDRINUSE:
                self.__logger.error("Unable to connect data socket to port {}"\
                             .format(command_port))
                raise e


        # Get keyboard emulator
        self.__keyboard = Controller()


    def start(self):
        """
        Start main event loop of the zmq kernel
        """

        self.start_game()

        try:
            self.__logger.debug("Controller Starting") 
            while True:
                self.__logger.debug("Requesting Screen")
                self.request_screen()

                self.__logger.debug("Waiting for game state data")
                try:
                    pkled_data = self.__data_input_socket.recv()
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        self.__logger.debug("Recv timeout")
                        continue

                self.__logger.debug("Recieved game state data")
                gm_st_dict = pickle.loads(pkled_data)
                self.do_control(gm_st_dict)
            
                
        except KeyboardInterrupt:
            pass # Fall through to quit

        self.quit()  

    def do_control(self, gm_st_dict):
        pass
        

    def get_zmq_context(self):
        """
        Return zmq context.
        """
        return self.__context
    
    def start_game(self):

        # Change cwd to mame
        os.chdir(MAME_PATH)

        program = os.path.join(MAME_PATH,"mame64")
        args = [program]
        args.append("-window")
        args.append("pacman")
        proc_mame = subprocess.Popen(args,stdin=subprocess.PIPE)

        # press keys to start
        sleep(0.5)
        self.__keyboard.press(Key.enter)
        sleep(0.25)
        self.__keyboard.release(Key.enter)
        sleep(3)
        self.__keyboard.press('5')
        sleep(0.25)
        self.__keyboard.release('5')
        sleep(2)
        self.__keyboard.press('5')
        sleep(0.25)
        self.__keyboard.release('5')
        sleep(0.5)
        self.__keyboard.press('1')
        sleep(0.25)
        self.__keyboard.release('1')
        sleep(3.8)

    def request_screen(self):

        sleep(0.1)
        self.__keyboard.press(Key.f12)
        sleep(0.25)
        self.__keyboard.release(Key.f12)
          
    def quit(self):
        """
        Shut down server
        """
        self.__logger.info("Initiating server shutdown")

        # Must close all sockets before context will terminate
        self.__data_input_socket.close()
        self.__context.term() 



if __name__ == "__main__":

    controller = ArcadeController(1111, DEBUG) 
    controller.start()


