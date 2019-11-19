import numpy as np
import cv2
from pacman import PAC_STATE



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
        self.nn   = node_number
        self.cost = cost

    def get_tuple(self):
        """
        Return tuple compatible with priority queue
        """
        return (self.cost, self) 

class Best_First_Searcher(object):
    def __init__(self, i_max):
        self.Q    = MinPriorityQueue()
        self.pred = []
        self.cost = []
        self.i_max = i_max 

    def run(self, x_0, y_0):
        alpha = 10 # Step size
        i     = 0  # Iteration number
        k     = 0  # Node number

        # Create initial node and add to queue and cost list
        node = Node(x_0, y_0, 0, k) 
        k += 1

        heappush(node.get_tuple())
        self.cost[node.node_number] = node.cost


        while(i in range(i_max)): 
            node_k = self.pop_Q() 

            # Get neighbors
            angles = [0, np.pi/2, np.pi, 3*np.pi/2]
            for theta in angles:
                delta_x = alpha*cos(theta)
                delta_y = alpha*sin(theta)
                new_pos = current_node.pos + [delta_x, delta_y]

                # Check if this move is valid

                node_k_1 = Node(new_pos[0], new_pos[1], cost, k)
                k += 1
 
                self.push_Q(node_k_1)
                self.prev[node_k_1] = node_k

            i = i+1







    def pop_Q(self):
        (cost, node) = heappop(self.Q)
        return node

    def push_Q(self, node):
        heappush(node.cost, node)

    def clear(self):
        self.Q.clear()
        self.pred = []
        self.cost = []






