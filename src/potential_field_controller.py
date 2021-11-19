import os

import numpy as np
import cv2

from controller import PacManController 

# Path to mame file system
MAME_PATH = os.environ["MAME_PATH"]



class PotentialFieldController(PacManController):
    def __init__(self, keyboard):
        super(PotentialFieldController, self).__init__(keyboard)

        self.__k = 0


    def initialize(self):

        # Remove all saved potential images
        os.system("rm {}/*.png".format(MAME_PATH))

    def compute_control(self, game_state):

        self.__k = game_state["k"]

        # Potential map should be uint8 image
        potential_map = game_state["potential_map"]



        p_y, p_x = game_state["PACMAN"][0], game_state["PACMAN"][1]
        curr_pos = np.array([p_x, p_y], dtype=np.int)
        self.logger.debug("PacPos: (x: {}, y: {}".format(p_x, p_y))  

        dx, dy   = self.get_gradient(p_x, p_y, potential_map)

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


        self.logger.debug("Grad : (dx: {}, dy: {})".format(grad_pos[0], grad_pos[1]))
        self.logger.debug("Next Pos: (x: {},y: {})".format(next_pos[0], next_pos[1]))
        self.logger.debug("Theta: {}".format(np.rad2deg(theta)))

        if(True in np.isnan(next_pos )):
            # There was no gradient (Likely that Ghosts are blue) 
            return 
        else:
            self.draw_gradient(potential_map, curr_pos, next_pos)


        if(theta > 0 and theta <= np.pi/2):
            self.logger.debug("RIGHT UP") 
            self.send_right()
            self.send_up()
        elif(theta > np.pi/2 and theta <= np.pi):
            self.logger.debug("LEFT UP") 
            self.send_left()
            self.send_up()
        elif(theta > np.pi and theta <= 3*np.pi/2):
            self.logger.debug("LEFT DOWN") 
            self.send_left()
            self.send_down()
        else:
            self.logger.debug("RIGHT DOWN") 
            self.send_right()
            self.send_down()
        

    def draw_gradient(self, potential_map, curr_pos, next_pos):
        #self.logger.debug("Drawing gradient")


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
        self.logger.debug("Wrote image : {}".format(image_name))

        check = cv2.imwrite(savepath, potential_map) 
        if(check == False):
           self.logger.warning("Did not save gradient image")

        #cv2.imshow("grad_img", potential_map)
        #cv2.waitKey(10) 



    def get_gradient(self, x_pos, y_pos, potential_map):

        height = potential_map.shape[0]
        width = potential_map.shape[1]

        # Handle boundary conditions
        h = 5
        if(x_pos - h <= 0):
            f_x1 = potential_map[y_pos][x_pos]
            f_x2 = potential_map[y_pos][x_pos+h]
            den  = h 
        elif(x_pos + h >= width):
            f_x1 = potential_map[y_pos][x_pos-h]
            f_x2 = potential_map[y_pos][x_pos]
            den  = h
        else:
            f_x1 = potential_map[y_pos][x_pos-h]
            f_x2 = potential_map[y_pos][x_pos+h]
            den  = 2*h
        dx   = (float(f_x2) - float(f_x1)) / den
        #self.logger.debug("fx1 {} fx2 {}".format(f_x1, f_x2))
        #self.logger.debug("dx {}".format(dx))

        # Handle boundary conditions
        if(y_pos - h <= 0):
            f_y1 = potential_map[y_pos][x_pos]
            f_y2 = potential_map[y_pos+h][x_pos]
            den  = h
        elif(y_pos + h >= height):
            f_y1 = potential_map[y_pos][x_pos]
            f_y2 = potential_map[y_pos-h][x_pos]
            den  = h
        else:
            f_y1 = potential_map[y_pos-h][x_pos]
            f_y2 = potential_map[y_pos+h][x_pos]
            den  = 2*h 
        dy   = (float(f_y2) - float(f_y1)) / den 
        #self.logger.debug("fy1 {} fy2 {}".format(f_y1, f_y2))
        #self.logger.debug("dy {}".format(dy))

        return dx, dy
