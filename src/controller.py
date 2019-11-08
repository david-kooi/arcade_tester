import os
import zmq
import pickle
import subprocess
from time import sleep
from pynput.keyboard import Key, Controller
import pynput.keyboard as keyboard

import numpy as np
import cv2

from threading import Thread

from zmq.eventloop.ioloop import IOLoop, PeriodicCallback 
from zmq.eventloop.zmqstream import ZMQStream

from zmq.eventloop.ioloop import IOLoop, PeriodicCallback 
from zmq.eventloop.zmqstream import ZMQStream

from logging import DEBUG, INFO, ERROR 
from utils import logging_util
from utils.logging_util import GetLogger

from state_estimator import normalize
from compute_potential import get_gradient, compute_potential,assign_potential_value 




# Path to arcade tester project root
AT_ROOT   = os.environ["AT_ROOT"]
LOG_PATH  = os.path.join(AT_ROOT, "logs")

# Path to mame file system
MAME_PATH = os.environ["MAME_PATH"]

WHITE = (0, 255, 255)

PAC_STATE = {}
PAC_STATE["EAT"] = 0

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
        self.__k = 0
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


        self.__pac_state = PAC_STATE["EAT"]

        self.control_timer = ControlTimer(0.05, self.notify_control_ready) 
        self.keyboard_listener = keyboard.Listener(on_press=self.notify_keypress)


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


        # Start keyboard listener
        #self.keyboard_listener.start()

    def start(self):
        """
        Start main event loop of the zmq kernel
        """

        self.start_game()
        self.control_timer.start()

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
                        self.__logger.warning("Game State Data Recv timeout")
                        continue

                game_state = pickle.loads(pkled_data)

                self.__logger.debug("Recieved game state data " + game_state["k"])
                self.__k = game_state["k"]
 
                try:
                    self.do_control(game_state)
                except Exception as e:
                    self.__logger.error(e)
                    self.quit()

            
                
        except KeyboardInterrupt:
            pass # Fall through to quit

        self.quit()  

    def notify_control_ready(self):
        self.control_ready = True
    def notify_keypress(self, key):
	print("Recieved: {}".format(key))

    def do_control(self, game_state):

        img_height = game_state["img_height"] 
        img_width = game_state["img_width"]

        # Potential map should be uint8 image
        potential_map = compute_potential(img_height, img_width,\
                                          game_state, self.__pac_state)


        p_y, p_x = game_state["PACMAN"][0], game_state["PACMAN"][1]
        curr_pos = np.array([p_x, p_y], dtype=np.int)
        self.__logger.debug("PacPos: (x: {}, y: {}".format(p_x, p_y))  

        dx, dy   = get_gradient(p_x, p_y, potential_map)

        # Compute next step

        grad_pos = np.array([dx, dy], dtype=np.float) 
        grad_pos = (grad_pos / np.linalg.norm(grad_pos)) # Normalize direction

        alpha = 30
        next_pos = curr_pos - alpha * grad_pos 

        # Get angle between horizontal and negative gradient
        neg_grad = -grad_pos
        unit_x   = np.array([1,0])
        inner    = np.dot(unit_x, neg_grad) / \
                ( np.linalg.norm(neg_grad) * np.linalg.norm(unit_x))
        theta = np.arccos(inner) 

        # If grad_y is positive, direction is negative
        # and we need to flip degrees 
        if(grad_pos[1] < 0):
            theta = 2*np.pi - theta
 
        self.__logger.debug("Grad : (dx: {}, dy: {})".format(grad_pos[0], grad_pos[1]))
        self.__logger.debug("Next Pos: (x: {},y: {})".format(next_pos[0], next_pos[1]))
        self.__logger.debug("Theta: {}".format(np.rad2deg(theta)))

        if(True in np.isnan(next_pos )):
            # There was no gradient (Likely that Ghosts are blue) 
            return 
        else:
            self.draw_gradient(potential_map, curr_pos, next_pos)

         

        if(self.control_ready):
            self.control_ready = False
        
            #self.one_key_control(theta)
            self.two_key_control(theta)
        else:
            self.__logger.debug("Control not ready")


    def one_key_control(self, theta):
        
        pi = np.pi

        if(theta > 0 and theta <= pi/4 or theta > 6*pi/4):
            self.__logger.debug("RIGHT") 
            self.send_right()
        elif(theta > pi/4 and theta <= 3*pi/4):
            self.__logger.debug("UP") 
            self.send_up()
        elif(theta > 3*pi/4 and theta <= 5*pi/4):
            self.__logger.debug("LEFT") 
            self.send_left()
        else:
            self.__logger.debug("DOWN") 
            self.send_down()


    def two_key_control(self, theta):

        pi = np.pi

        if(theta > 0 and theta <= pi/2):
            self.__logger.debug("RIGHT UP") 
            self.send_right()
            self.send_up()
            self.prev_cmd = "RU"
        elif(theta > pi/2 and theta <= np.pi):
            self.__logger.debug("LEFT UP") 
            self.send_left()
            self.send_up()
            self.prev_cmd = "LU"
        elif(theta > pi and theta <= 3*np.pi/2):
            self.__logger.debug("LEFT DOWN") 
            self.send_left()
            self.send_down()
            self.prev_cmd = "LD"
        else:
            self.__logger.debug("RIGHT DOWN") 
            self.send_right()
            self.send_down()
            self.prev_cmd = "RD"


    def draw_gradient(self, potential_map, curr_pos, next_pos):
        #self.__logger.debug("Drawing gradient")


        # Add pacman to image 
        px = int(curr_pos[0])
        py = int(curr_pos[1])
        cv2.circle(potential_map, (px, py), 8, 255, -1) 

        # Add gradient direction to image
        pt1 = int(curr_pos[0]), int(curr_pos[1])
        pt2 = int(next_pos[0]), int(next_pos[1])
        cv2.line(potential_map, pt1, pt2, 255, 2) 

        ## Write image to file
        image_name = "potential_img_" + str(self.__k) + ".png"
        savepath = os.path.join(os.getcwd(),image_name)
        self.__logger.debug("Wrote image : {}".format(image_name))

        check = cv2.imwrite(savepath, potential_map) 
        if(check == False):
           self.__logger.warning("Did not save gradient image")

        #cv2.imshow("grad_img", potential_map)
        #cv2.waitKey(10) 




# Old Method
#        grad = np.gradient(potential_map) 
#        x_grad = grad[0] 
#        y_grad = grad[1]
#
#        ## Write image to file
#        savepath = os.path.join(os.getcwd(),"grad_img_" + str(self.__k) + ".png")
#        cv2.imwrite(savepath, x_grad)
#

#        return x_grad[x_pos][y_pos], y_grad[x_pos][y_pos] 

        

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
        #args.append("-sound 'none'") # Unsure why this does not work
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
        self.press_key(Key.f12)

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
        sleep(0.05)
        self.__keyboard.release(key)


    def quit(self):
        """
        Shut down server
        """
        self.__logger.info("Initiating server shutdown")

        # Must close all sockets before context will terminate
        self.__data_input_socket.close()
        self.__context.term() 

        # Stop timer
        self.control_timer.on = False

class ControlTimer(Thread):
    def __init__(self, t, callback):
        Thread.__init__(self)

        self.t = t
        self.callback = callback
        self.on = True 

    def run(self):
        while self.on: 
            sleep(self.t)
            self.callback()

#class KeyboardListener(Thread):
#    def __init__(self):
#        Thread.__init__(self)
#
#    def run(self):
#



    
if __name__ == "__main__":

    # Remove all saved potential images
    os.system("rm {}/*.png".format(MAME_PATH))


    controller = ArcadeController(1111, DEBUG) 
    controller.start()


