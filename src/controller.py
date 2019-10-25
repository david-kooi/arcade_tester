import os
import zmq
import pickle
import subprocess
from time import sleep
from pynput.keyboard import Key, Controller

import numpy as np
import cv2

from zmq.eventloop.ioloop import IOLoop, PeriodicCallback 
from zmq.eventloop.zmqstream import ZMQStream

from zmq.eventloop.ioloop import IOLoop, PeriodicCallback 
from zmq.eventloop.zmqstream import ZMQStream

from logging import DEBUG, INFO, ERROR 
from utils import logging_util
from utils.logging_util import GetLogger

from state_estimator import normalize




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
                game_state = pickle.loads(pkled_data)
                self.do_control(game_state)
            
                
        except KeyboardInterrupt:
            pass # Fall through to quit

        self.quit()  

    def do_control(self, game_state):
        potential_map = game_state["potential_map"]

        p_x, p_y = game_state["PACMAN"][0], game_state["PACMAN"][1]
        dx, dy   = self.get_gradient(p_x, p_y, potential_map)

        # Compute next step
        curr_pos = np.array([p_x, p_y])
        grad_pos = np.array([dx, dy])

        alpha = 1
        next_pos = curr_pos - alpha * grad_pos 

        # Get angle to next pos
        delta_pos = next_pos - curr_pos
        theta = np.arctan(delta_pos[1]/delta_pos[0])

        self.draw_gradient(curr_pos, grad_pos)

        if(theta > 0 and theta <= np.pi/2):
            pass
        elif(theta > np.pi/2 and theta <= np.pi):
            pass
        elif(theta > np.pi and theta <= 3*np.pi/2):
            pass
        else:
            pass



    def draw_gradient(self, curr_pos, grad_pos):
       grad_img = np.zeros((100, 100), np.uint8) 
       grad_img.fill(255)

       next_pos = curr_pos + grad_pos
       pt1 = int(curr_pos[0]), int(curr_pos[1])
       pt2 = int(next_pos[0]), int(next_pos[1])

       cv2.line(grad_img, pt1, pt2, 0) 
       cv2.imshow("grad_img", grad_img)
       cv2.waitKey(10) 


        
    def get_gradient(self, x_pos, y_pos, potential_map):
        grad = np.gradient(potential_map)

        x_grad = grad[0] 
        y_grad = grad[1]

        return x_grad[x_pos][y_pos], y_grad[x_pos][y_pos] 

        

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


