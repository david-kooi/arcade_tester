import cv2
import numpy as np
import zmq
import time
import pickle
import sys
import os
import glob
import struct

# Path to mame file system
MAME_PATH = os.environ["MAME_PATH"]
SNAP_PATH = os.path.join(MAME_PATH,"snap/pacman")

# Sprite colors in HSV
PAC_YELLOW      = (60, 255, 255)
GH_RED          = (0.25, 255, .949*255)
GH_BLUE         = (180.24, .988*255, .9765*255)
GH_ORANGE       = (32.69, .5847*255, .9725*255)
GH_PINK         = (301.85, .261*255, .9765*255)
GH_WHITE        = (240, .13*255, .99*255)
PILL_RED        = (9, .3175*255, .9882*255)
BORDER_BLUE     = (240, .871*255, 1*255)
BORDER_BLUE_BGR = (255, 33, 33)
COLOR_LIST      = [PAC_YELLOW, GH_RED, GH_BLUE, GH_ORANGE, GH_PINK]


# mid_first_pill[0]: Row position
# mid_first_pill[1]: Row position
MID_FIRST_PILL = [0, 0]
#MID_LAST_PILL = [0, 0]

# PILL_DIST[0]: Row spacing
# PILL_DIST[1]: Column spacing
PILL_DIST = [0, 0]

global FIRST_RUN
FIRST_RUN = True

PINK_THICK = 2
PINK_BLOCK_LEFT  = (103, 125)
PINK_BLOCK_RIGHT = (120, 125)
TOP_THICK = 23
TOP_BLOCK_LEFT  = (0, 11)
TOP_BLOCK_RIGHT = (224, 11)
BOTTOM_THICK = 16
BOTTOM_BLOCK_LEFT  = (0, 280)
BOTTOM_BLOCK_RIGHT = (224, 280)



def norm(peak_x, peak_y, mesh_x, mesh_y):
    C_squared = (peak_x - mesh_x)**2 + (peak_y - mesh_y)**2 

    return np.sqrt(C_squared)

def normalize(X, x_min, x_max):
    diff = x_max - x_min
    num  = X - x_min
    den  = np.max(X) - np.min(X)
    return (diff*num/den) + x_min 

def block_unwanted(img_bgr):
    cv2.line(img_bgr, PINK_BLOCK_LEFT, PINK_BLOCK_RIGHT, (255, 33, 33), PINK_THICK)
    cv2.line(img_bgr, TOP_BLOCK_LEFT, TOP_BLOCK_RIGHT, (0, 0, 0), TOP_THICK)
    cv2.line(img_bgr, BOTTOM_BLOCK_LEFT, BOTTOM_BLOCK_RIGHT, (0, 0, 0), BOTTOM_THICK)

def process_image(img_bgr, game_state):
    """
    1. Add average color positions of characters in COLOR_LIST to game_state
    2. Find pill location and add to game state
    3. Compute potential field 
    """
    process_pills(img_bgr, game_state) 
    isolate_characters(img_bgr, game_state) 

    game_state["img_height"] =  img_bgr.shape[0]
    game_state["img_width"] = img_bgr.shape[1]    
    #compute_potential(img_bgr, game_state)

def get_border(img_hsv, game_state):
    H_lo = BORDER_BLUE[0]/2 - 10
    H_hi = BORDER_BLUE[0]/2 + 10

    S_lo = BORDER_BLUE[1] - 30
    S_hi = BORDER_BLUE[1] + 30 

    V_lo = BORDER_BLUE[2] - 30 
    V_hi = BORDER_BLUE[2] + 30 

    lower_bound = (H_lo, S_lo, V_lo) 
    upper_bound = (H_hi, S_hi, V_hi)

    result = cv2.inRange(img_hsv, lower_bound, upper_bound)
    _, contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    game_state["border"] = contours

def check_dark_ghosts(img_hsv, game_state):
    H_lo = BORDER_BLUE[0]/2 - 10
    H_hi = BORDER_BLUE[0]/2 + 10

    S_lo = BORDER_BLUE[1] - 30
    S_hi = BORDER_BLUE[1] + 30

    V_lo = BORDER_BLUE[2] - 30
    V_hi = BORDER_BLUE[2] + 30

    lower_bound = (H_lo, S_lo, V_lo)
    upper_bound = (H_hi, S_hi, V_hi)

    result = cv2.inRange(img_hsv, lower_bound, upper_bound)

    for contour in game_state["border"]:
        cv2.fillPoly(result, contour, (0, 0, 0))

    _, contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    game_state["GHDARK"] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 4 and area < 1000:
            moment = cv2.moments(contour)
            cX = int(moment["m10"] / moment["m00"])
            cY = int(moment["m01"] / moment["m00"])
            game_state["GHDARK"].append((cY, cX))


    H_lo = GH_WHITE[0]/2 - 5
    H_hi = GH_WHITE[0]/2 + 5

    S_lo = GH_WHITE[1] - 15
    S_hi = GH_WHITE[1] + 15

    V_lo = GH_WHITE[2] - 10
    V_hi = GH_WHITE[2] + 10

    lower_bound = (H_lo, S_lo, V_lo)
    upper_bound = (H_hi, S_hi, V_hi)

    result2 = cv2.inRange(img_hsv, lower_bound, upper_bound)
    _, contours, _ = cv2.findContours(result2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 12:
            moment = cv2.moments(contour)
            cX = int(moment["m10"] / moment["m00"])
            cY = int(moment["m01"] / moment["m00"])
            game_state["GHDARK"].append((cY, cX))

def isolate_characters(img_bgr, game_state):
    """
    @input img: Loaded BGR image 
    @input game_state: Game state dictionary
    
    Isolates each color in COLOR_LIST and determines average position.
    Adds the position to the game_state dictionary.
    """

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    positions = []
    for color in COLOR_LIST:
        lower_bound = ((color[0]-30)/2, color[1]-30, color[2]-30)
        upper_bound = ((color[0]+30)/2, color[1]+30, color[2]+30)
        mask0 = cv2.inRange(img_hsv, lower_bound, upper_bound)
        x_output, y_output = get_avg_pos(mask0)
        positions.append((x_output, y_output))

    game_state["PACMAN"] = positions[0]
    game_state["GHRED"] = positions[1]
    game_state["GHBLUE"] = positions[2]
    game_state["GHORANGE"] = positions[3]
    game_state["GHPINK"] = positions[4]

    check_dark_ghosts(img_hsv, game_state)


def get_avg_pos(isolated_image):
    """
    @input hsv_img: HSV type image
    @input COLOR_LIST: color list to isolate sprites
    @returns: list of xy values of sprite locations
    """

    x_list, y_list = np.where(isolated_image == 255)
    try:
        avg_x = sum(x_list) / len(x_list)
        avg_y = sum(y_list) / len(y_list)
        return avg_x, avg_y
    except:
        return 0, 0

def determine_pill_grid(pill_px):
    

    # get mid-point of first and last pill
    MID_FIRST_PILL[0] = pill_px[0][0]
    MID_FIRST_PILL[1] = pill_px[1][0]
    for i in range(0, len(pill_px[1])-1):
        if pill_px[1][i + 1] - pill_px[1][i] > 1:
            #MID_FIRST_PILL[1] = (pill_px[1][i] + pill_px[1][0]) / 2
            PILL_DIST[1] = pill_px[1][i + 1] - pill_px[1][0]
            break
    #for i in range(1, len(pill_px[1])):
    #    if pill_px[1][len(pill_px[1]) - i] - pill_px[1][len(pill_px[1]) - i - 1] > 1:
    #        MID_LAST_PILL[1] = (pill_px[1][len(pill_px[1]) - i] + pill_px[1][len(pill_px[1]) - 1]) / 2
    #        break
    for j in range(0, len(pill_px[0])-1):
        if pill_px[0][j + 1] - pill_px[0][j] > 1:
            #MID_FIRST_PILL[0] = (pill_px[0][j] + pill_px[0][0]) / 2
            PILL_DIST[0] = pill_px[0][j + 1] - pill_px[0][0]
            break
    #for j in range(1, len(pill_px[0])):
    #    if pill_px[0][len(pill_px[0]) - j] - pill_px[0][len(pill_px[0]) - j - 1] >= 2:
    #        MID_LAST_PILL[0] = (pill_px[0][len(pill_px[0]) - j] + pill_px[0][len(pill_px[0]) - 1]) / 2
    #        break


def process_pills(img_bgr, game_state):
    """
    @input img_bgr: BGR type image 
    @input game_state: Dictionary of game state
    @returns: xy values of red pills
    """
    global FIRST_RUN

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(img_hsv, (0, .2 * 255, .9 * 255), (20, .4 * 255, 255))

    if(FIRST_RUN):
        ##TODO: Better organize
        get_border(img_hsv, game_state)

        pill_px = np.where(img_threshold == 255)
        try:
            determine_pill_grid(pill_px)
            FIRST_RUN = False
        except IndexError:
            FIRST_RUN = True


    game_state["small_pills"] = []
    game_state["big_pills"] = []

    # check if pill is at intersection
    try:
        for j in range(0, 26):
            for i in range(0, 29):
                if np.any(img_threshold[MID_FIRST_PILL[0] + i * PILL_DIST[0], MID_FIRST_PILL[1] + j * PILL_DIST[1]] == 255):
                    if (i == 2 and (j == 0 or j == 25)) or (i == 22 and (j == 0 or j == 25)):
                        game_state["big_pills"].append(
                            (MID_FIRST_PILL[0] + i * PILL_DIST[0], MID_FIRST_PILL[1] + j * PILL_DIST[1]))
                    else:
                        game_state["small_pills"].append(
                            (MID_FIRST_PILL[0] + i * PILL_DIST[0], MID_FIRST_PILL[1] + j * PILL_DIST[1]))
    except IndexError:
        pass

def compute_potential(bgr_img, game_state):

    img_height = bgr_img.shape[0]
    img_width  = bgr_img.shape[1]


    # Create the x, y coordinates 
    #potential_map = np.zeros((img_width, img_height,1),np.int8) 
    xx = np.linspace(0, img_width, img_width)
    yy = np.linspace(0, img_height, img_height)
    X, Y = np.meshgrid(xx, yy)

    Z = X.copy() # Potential values 
    D = X.copy() # Distance values
    Z.fill(0)
    D.fill(0)
    


    # Compute positive potential for ghosts
    ghost_parameters = {"Ca": 400, "Cb":200}
    ghost_positions = []
    ghost_positions.append( game_state["GHRED"])
    ghost_positions.append( game_state["GHBLUE"])
    ghost_positions.append( game_state["GHORANGE"])
    ghost_positions.append( game_state["GHPINK"])


    assign_potential_value(game_state,\
            ghost_parameters,\
            ghost_positions, \
            X,Y,Z)

    # Compute negative potential for small pills
    s_pill_parameters = {"Ca": -10, "Cb":10}
    s_pill_positions = game_state["small_pills"]

    assign_potential_value(game_state, \
            s_pill_parameters, \
            s_pill_positions, \
            X,Y,Z)


    # Plot 2D
    #fig_2d = plt.figure()
    #ax = fig_2d.gca()

    Z_2d = normalize(Z, 0, 255)
    Z_2d += 128
    Z_2d = np.clip(Z_2d, 0, 255)
    Z_2d = Z_2d.astype(np.uint8) 

    # Package to send to controller
    game_state["potential_map"] = Z_2d.copy() 


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

def assign_potential_value(game_state, parameters, positions, X,Y,Z):

    # Potential function parameters
    Ca = parameters["Ca"]
    Cb = parameters["Cb"] 

    # Create a potential map
    # Note: Positions are flipped in images 
    for (y,x) in positions:

        # Get norm of all points away from sprite 
        D = norm(x,y, X,Y)
        D = D + 0.001 # Get rid of zero

        # Calculate potential
        Z_i = Ca / (D**2 + Cb)
        Z += Z_i 
    



def draw_track(img, game_state):
  
    character_size  = 8 
    small_pill_size = 2 
    big_pill_size   = 6 

    height = img.shape[0]
    width  = img.shape[1]

    new_img = np.zeros((height, width, 3), np.uint8)

    for contour in game_state["border"]:
        area = cv2.contourArea(contour)
        if area != 1.0 and (area < 100.0 or area > 300.0 or area == 153.0 or area == 209.0):
            cv2.drawContours(new_img, contour, -1, (255, 33, 33), 1)

    cv2.circle(new_img, (game_state["PACMAN"][1], game_state["PACMAN"][0]), character_size, (0, 255, 255))
    cv2.circle(new_img, (game_state["GHRED"][1], game_state["GHRED"][0]), character_size, (0, 0, 255))
    cv2.circle(new_img, (game_state["GHBLUE"][1], game_state["GHBLUE"][0]), character_size, (255, 255, 0))
    cv2.circle(new_img, (game_state["GHORANGE"][1], game_state["GHORANGE"][0]), character_size, (75, 170, 233))
    cv2.circle(new_img, (game_state["GHPINK"][1], game_state["GHPINK"][0]), character_size, (255, 185, 255))
    for ghost in game_state["GHDARK"]:
        cv2.circle(new_img, (ghost[1], ghost[0]), character_size, (255, 33, 33))
    for pill in game_state["small_pills"]:
        cv2.circle(new_img, (pill[1], pill[0]), small_pill_size, (255, 255, 255))
    for pill in game_state["big_pills"]:
        cv2.circle(new_img, (pill[1], pill[0]), big_pill_size, (255, 255, 255))


    #final = cv2.resize(new_img, (int(.5*1126), int(.5*1275)))
    #ret,gray = cv2.threshold(final, 0, (255, 0, 0), cv2.THRESH_BINARY)

    new_img = np.expand_dims(new_img, axis=-1)

    width = width * 2
    height = height * 2
    resized = cv2.resize(new_img, (width, height))
        
    cv2.imshow("newtrack", resized)
    cv2.waitKey(10)


def setup_zmq(port):
    """
    Setup state_estimator as publisher
    """

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:1111")

    return socket

def get_latest_file(folder):
    """
    Get the most recent file in folder
    """

    list_of_files = glob.glob(os.path.join(folder, "*"))
    if(not list_of_files):
        raise Exception

    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def main(port):

    game_state = {}
    prev_filepath= ""

    socket = setup_zmq(port)

    # Swipe all previous snapshots
    os.system("rm {}/*".format(SNAP_PATH))
 
    k = 0
    while True:
       
        try:
            latest_filename = get_latest_file(SNAP_PATH)
            latest_filepath = os.path.join(SNAP_PATH, latest_filename) 
        except KeyboardInterrupt:
            return
        except:
            continue


        if(prev_filepath == latest_filepath):
           pass 
        else: 

            latest_img_bgr = cv2.imread(latest_filepath)
            if(latest_img_bgr is None):
                continue

            print(latest_filepath)

            block_unwanted(latest_img_bgr)
     
            process_image(latest_img_bgr, game_state)
            

            game_state["k"] = str(k)
            k += 1
            pkled_data = pickle.dumps(game_state) 

            socket.send(pkled_data)
            #draw_track(latest_img_bgr, game_state)

            prev_filepath = latest_filepath
            
        




if __name__ == "__main__":

    port = 1111
    main(1111)
