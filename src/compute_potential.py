import os
from heapq import * 

import numpy as np
import cv2

from logging import DEBUG, INFO, ERROR 
from utils import logging_util
from utils.logging_util import GetLogger

from pacman import PAC_STATE


# Path to arcade tester project root
AT_ROOT   = os.environ["AT_ROOT"]
LOG_PATH  = os.path.join(AT_ROOT, "logs")


def norm(peak_x, peak_y, mesh_x, mesh_y):
    C_squared = (peak_x - mesh_x)**2 + (peak_y - mesh_y)**2 

    return np.sqrt(C_squared)

def normalize_1(X, x_min, x_max):
    diff = x_max - x_min
    num  = X - x_min
    den  = np.max(X) - np.min(X)
    return (diff*num/den) + x_min 

def normalize(X, x_min, x_max):
    X += -(np.min(X))
    X /= np.max(X) / (x_max - x_min)
    X += x_min
    return X


def get_gradient(x_pos, y_pos, potential_map):

    height = potential_map.shape[0]
    width = potential_map.shape[1]

    # Handle boundary conditions
    h = 4
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
    #self.__logger.debug("fx1 {} fx2 {}".format(f_x1, f_x2))
    #self.__logger.debug("dx {}".format(dx))

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
    #self.__logger.debug("fy1 {} fy2 {}".format(f_y1, f_y2))
    #self.__logger.debug("dy {}".format(dy))

    return dx, dy



def compute_potential(controller, img_height, img_width, game_state):

    # Create the x, y grid 
    xx = np.linspace(0, img_width, img_width)
    yy = np.linspace(0, img_height, img_height)
    X, Y = np.meshgrid(xx, yy)

   
    D = X.copy() # Distance values
    D.fill(0)
    
    # Form barriers
    Vb = X.copy()
    Vb.fill(0)
    cv2.drawContours(Vb, game_state['border'], -1, 255, -1)
    Vb = Vb.astype(np.uint8)



    ## Compute potential for ghosts
    ghost_positions = []
    ghost_positions.append( game_state["GHRED"])
    ghost_positions.append( game_state["GHBLUE"])
    ghost_positions.append( game_state["GHORANGE"])
    ghost_positions.append( game_state["GHPINK"])
    # Ghost Parameters
    Mg      = 100 
    alpha_g = 0.075

    Vg = X.copy() # Ghost potential values 
    Vg.fill(0)
    def Vg_i(D):
        # Assuming all elements of D are positive

        # Crop D to the domain of the ghost
        Dg   = np.sqrt(2*Mg/alpha_g)
        Vg_i = Mg - 0.5*(alpha_g) * D**2
        Vg_i = np.where(D < Dg, Vg_i, 0) 
    
        return Vg_i

    assign_potential_value(Vg_i, ghost_positions, X,Y,Vg)


    # Compute potentials for small pills
    Mp =20 
    alpha_p = 0.1
    s_pill_positions = game_state["small_pills"]

    Vp = X.copy() 
    Vp.fill(Mp) 
    def Vp_i(D):
        # Assuming all elements of D are positive

        # Crop D to the domain of the pills 
        Dg   = np.sqrt(2*Mp/alpha_p)
        Vp_i = - Mp + 0.5*(alpha_p) * D**2
        Vp_i = np.where(D < Dg, Vp_i, 0) 
    
        return Vp_i

    assign_potential_value(Vp_i, s_pill_positions, X,Y,Vp) 




#    assign_potential_value(game_state, \
#            s_pill_parameters, \
#            s_pill_positions, \
#            X,Y,Vp)
#
    # Compute negative potential for vunerable ghosts
#    dark_ghost_parameters = {"Ca": -400, "Cb":400}
#    dark_ghost_positions  = game_state{"GHDARK"]
#    assign_potential_value(game_state,\
#            dark_ghost_parametersi,\
#            dark_ghost_positions,
#            X,Y,Vd)



    V = Vb + Vg + Vp + 50
    V = np.clip(V, 0, 255)

    V = V.astype(np.uint8) 

    return V, Vb


    # Display
    #img_width = img_width * 2
    #img_height = img_height * 2
    #resized = cv2.resize(Z_2d, (img_width, img_height))
        
#    Z_2d = normalize(Z_2d, 0, 1) 
#    cv2.imshow("newtrack", Z_2d) 
#    cv2.waitKey(10)


    

    #draw_track(Z_2d, game_state)
    
#    cv2.imwrite("norm.png", D_2d)
   # cv2.imwrite("potential.png", Z_2d)

#    cv2.waitKey(10)
    #plt.imshow(Z_2d, cmap='gray', vmin=0, vmax=255)
    #plt.show()

def compute_switch(controller, Vp, Vg):
    e1 = 50
    e2 = 20

    pac_state   = controller.pac_state
    p_x = controller.p_x
    p_y = controller.p_y

    Vg_xy = Vg[p_y][p_x]

    if(pac_state == PAC_STATE["EAT"]): 
        if(Vg_xy > e1):
            controller.pac_state = PAC_STATE["RUN"]
            V = Vg
        else:
            V = Vp
    elif(pac_state == PAC_STATE["RUN"]):
        if(Vg_xy < e2):
            controller.pac_state = PAC_STATE["EAT"]
            V = Vp
        else: 
            V = Vg
    else:
        raise Exception("Undefined pac_state {}".format(pac_state))

    return V

def assign_potential_value(Vi, positions, X,Y,V):

    # Create the potential map
    # Note: Positions are flipped in images 
    for (y,x) in positions:

        # Skip non-identified sprites
        if((y,x) == (0,0)):
            continue

        # Get norm of all points away from sprite 
        D = norm(x,y, X,Y)
        #D = D + 0.001 # Get rid of zero

        # Calculate potential
        V += Vi(D)     


class Node(object):
    def __init__(self, x, y, cost, node_number):
        self.pos  = [x,y]
        self.node_number= node_number
        self.cost = cost
        self.prev = None 

    def get_xy(self):
        return (self.pos[0], self.pos[1])
    def get_tuple(self):
        """
        Return tuple compatible with priority queue
        """
        return (self.cost, self) 

class BestFirstSearcher(object):
    def __init__(self, i_max, logger):
        self.Q         = [] # Min priority queue 
        self.i_max     = i_max 
        self.prev      = [None] * self.i_max # "Preallocate array"
        self.node_dict = dict()
        self.visited   = []

        self.neighbor_mask = None
        self.image_set = False


        self.__logger = GetLogger("BFS", LOG_PATH, logLevel=DEBUG,\
                                               fileLevel=DEBUG)

    def set_image_size(self, img):

        # Masking image to check if a neighbor is valid
        self.neighbor_mask = img.copy()


    def collect_neighbors(self, node_k, V, Vb, k):
        alpha = 20 # Step size
        x_0 = node_k.pos[0]
        y_0 = node_k.pos[1]

        self.__logger.debug("Collecting neighbors for node {}, ({},{})".format(k, x_0,y_0))

        # Get neighbors
        self.neighbor_mask.fill(0)
        angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        for theta in angles:
            self.__logger.debug("Theta: {}".format(360*theta/(2*np.pi)))
            delta_x = alpha*np.cos(theta)
            delta_y = alpha*np.sin(theta)
            new_pos = node_k.pos + np.array([delta_x, delta_y])

            # Check if this move is valid
            x_1 = int(new_pos[0])
            y_1 = int(new_pos[1])
            self.__logger.debug("New Pos ({},{})".format(x_1, y_1))

            #cv2.line(self.neighbor_mask, (x_0, y_0), (x_1, y_1), 255)  
            #self.neighbor_mask = cv2.bitwise_and(self.neighbor_mask, Vb)

            # We have seen this already
            if((x_1,y_1) in self.visited):
                self.__logger.debug("VISTED")
                continue


            # Check if there is overlap with a border
            if(np.max(self.neighbor_mask) > 0):

                image_name = "neighbor_mask_img_" + str(k) + ".png"
                savepath = os.path.join(os.getcwd(),image_name)
                #self.__logger.debug("Wrote image : {}".format(image_name))

                check = cv2.imwrite(savepath, self.neighbor_mask) 
                #if(check == False):
                #   self.__logger.warning("Did not save neighbor_mask image")
                continue
                
            else:
                # Direction is good: Add node to queue 
                try:
                    cost = V[y_1][x_1] 
                    self.__logger.debug("Cost {}".format(cost))
                except IndexError:
                    # Next step is out of bounds
                    continue


                k += 1
                node_k_1 = Node(x_1, y_1, cost, k)
                self.__logger.debug("Node {} ({},{})) added. Cost {}".format(k, x_1, y_1, cost))
                
                if(node_k_1.node_number == self.i_max):
                    # The maxinum number of nodes have been evaluated
                    return (True, node_k.node_number) # Return the latest
                                                      # node processed 
                else:             
                    self.push_Q(node_k_1)
                    self.prev[node_k_1.node_number] = node_k.node_number 
                    node_k_1.prev = node_k
                    self.node_dict[node_k_1.node_number] = node_k_1
                    self.visited.append(node_k_1.get_xy())

        return (False, k)

    def run(self, x_0, y_0, V, Vb):
        i     = 0  # Iteration number
        k     = -1 # Node number

        # Create initial node and add to queue and cost list
        k += 1
        node = Node(x_0, y_0, 0, k) 


        self.push_Q(node)
        self.prev[node.node_number] = -1 
        self.node_dict[node.node_number] = node
        self.visited.append(node.get_xy())

        while(True): 

            # Break if we run out of nodes to pop
            try:
                node_k = self.pop_Q() 
            except IndexError:
                break

            (collection_finished, k) = self.collect_neighbors(node_k, V, Vb, k)

            if(collection_finished):
                # If we have collected i_max nodes we break
                break 
            else:
                # Otherwise we continue
                continue 


            
        # Do backtrace
        # node_trace will be ordered from final node to initial   
        final_node = self.node_dict[k] 
        node_trace = [final_node]
        while(k > 0):
            prev_k = self.prev[k]
            prev_node = self.node_dict[prev_k]

            node_trace.append(prev_node)
            k = prev_k 

      
        self.clear_data()
        return node_trace


    def pop_Q(self):
        (cost, node) = heappop(self.Q)
        return node

    def push_Q(self, node):
        heappush(self.Q, (node.cost, node))

    def clear_data(self):
        self.Q         = []
        self.prev      = [None] * self.i_max
        self.node_dict = dict()
        self.visited   = []
    


