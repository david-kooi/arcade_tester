import cv2
import numpy as np
import zmq
import time
import pickle
import sys

# Sprite colors in HSV
PAC_YELLOW = (60, 255, 255)
GH_RED = (0.25, 255, .949*255)
GH_BLUE = (180.24, .988*255, .9765*255)
GH_ORANGE = (32.69, .5847*255, .9725*255)
GH_PINK = (301.85, .261*255, .9765*255)
PILL_RED = (9, .3175*255, .9882*255)
BORDER_BLUE = (241.52, .7453*255, .4157*255)
COLOR_LIST = [PAC_YELLOW, GH_RED, GH_BLUE, GH_ORANGE, GH_PINK]

game_state = {}

def dictionary(image_path):
    sprite_positions = extract_COLORS(image_path)

    game_state["PACMAN"] = sprite_positions[0]
    game_state["GHRED"] = sprite_positions[1]
    game_state["GHBLUE"] = sprite_positions[2]
    game_state["GHORANGE"] = sprite_positions[3]
    game_state["GHPINK"] = sprite_positions[4]
    process_pills(image_path)


def extract_COLORS(image_path):
    img = cv2.imread(image_path)
    cv2.line(img, (522, 534), (600, 534), (255, 33, 33), 14)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    positions = []
    for color in COLOR_LIST:
        lower_bound = ((color[0]-30)/2, color[1]-30, color[2]-30)
        upper_bound = ((color[0]+30)/2, color[1]+30, color[2]+30)
        mask0 = cv2.inRange(img_hsv, lower_bound, upper_bound)
        x_output, y_output = get_avg_pos(mask0)
        positions.append((x_output, y_output))
    return positions

def get_img(img_path):
    """
    @input img_path: Full path of the image
    @returns: A BGR image of the image path 
    """
    pass
# General sprite isolation functions
def iso_color(color, hsv_img):
    """
    @input color: (H,S,V) color tuple to isolate
    @input hsv_img: HSV type image 
    @returns: Grayscale image of the isolated color
    """
    pass

def cvt_bgr_to_hsv(bgr_img):
    """
    @input bgr_img: BGR type image
    @returns: A HSV type image
    """

    pass




def get_avg_pos(isolated_image):
    """
    @input hsv_img: HSV type image
    @input COLOR_LIST: color list to isolate sprites
    @returns: list of xy values of sprite locations
    """

    x_list, y_list = np.where(isolated_image == 255)
    avg_x = sum(x_list) / len(x_list)
    avg_y = sum(y_list) / len(y_list)
    return avg_x, avg_y

def process_pills(image_path):
    """
    @input hsv_img: HSV type image
    @input PILL_RED: isolate red pills
    @returns: xy values of red pills
    """
    img = cv2.imread(image_path)
    cv2.line(img, (522, 534), (600, 534), (255, 33, 33), 14)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    result = cv2.inRange(hsv, (0, .2 * 255, .9 * 255), (20, .4 * 255, 255))
    height = hsv.shape[0]
    width = hsv.shape[1]
    pill_px = np.where(result == 255)
    mid_first_pill = [0, 0]
    mid_last_pill = [0, 0]
    pill_dist = [0, 0]
    # get mid-point of firet and last pill
    for i in range(0, len(pill_px[1])):
        if pill_px[1][i + 1] - pill_px[1][i] > 1:
            mid_first_pill[1] = (pill_px[1][i] + pill_px[1][0]) / 2
            pill_dist[1] = pill_px[1][i + 1] - pill_px[1][0]
            break
    for i in range(1, len(pill_px[1])):
        if pill_px[1][len(pill_px[1]) - i] - pill_px[1][len(pill_px[1]) - i - 1] > 1:
            mid_last_pill[1] = (pill_px[1][len(pill_px[1]) - i] + pill_px[1][len(pill_px[1]) - 1]) / 2
            break
    for j in range(0, len(pill_px[0])):
        if pill_px[0][j + 1] - pill_px[0][j] >= 2:
            mid_first_pill[0] = (pill_px[0][j] + pill_px[0][0]) / 2
            pill_dist[0] = pill_px[0][j + 1] - pill_px[0][0]
            break
    for j in range(1, len(pill_px[0])):
        if pill_px[0][len(pill_px[0]) - j] - pill_px[0][len(pill_px[0]) - j - 1] >= 2:
            mid_last_pill[0] = (pill_px[0][len(pill_px[0]) - j] + pill_px[0][len(pill_px[0]) - 1]) / 2
            break
    game_state["small_pills"] = [(0, 0)]
    game_state["big_pills"] = [(0, 0)]
    pill_list = {"small_pills": [(0, 0)], "big_pills": [(0, 0)]}
    # check if pill is at intersection
    for j in range(0, 26):
        for i in range(0, 15):
            if np.any(result[mid_first_pill[0] + i * pill_dist[0], mid_first_pill[1] + j * pill_dist[1]] == 255):
                if i == 2 and (j == 0 or j == 25):
                    game_state["big_pills"].append(
                        (mid_first_pill[0] + i * pill_dist[0], mid_first_pill[1] + j * pill_dist[1]))
                else:
                    game_state["small_pills"].append(
                        (mid_first_pill[0] + i * pill_dist[0], mid_first_pill[1] + j * pill_dist[1]))
        for i in range(0, 14):
            if np.any(result[mid_last_pill[0] - i * pill_dist[0], mid_first_pill[1] + j * pill_dist[1]] == 255):
                if i == 6 and (j == 0 or j == 25):
                    game_state["big_pills"].append(
                        (mid_last_pill[0] - i * pill_dist[0], mid_first_pill[1] + j * pill_dist[1]))
                else:
                    game_state["small_pills"].append(
                        (mid_last_pill[0] - i * pill_dist[0], mid_first_pill[1] + j * pill_dist[1]))
    del game_state["small_pills"][0]
    del game_state["big_pills"][0]

def process_obs(BORDER_BLUE, hsv_img):
    """
    @input hsv_img: HSV type image
    @input BORDER_BLUE: isolate blue border 
    @returns: processes locations of blue border
    """
    pass

def draw_track(image_path):
    img = cv2. imread(image_path)
    cv2.line(img, (522, 534), (600, 534), (255, 33, 33), 14)
    new_img = np.zeros((1275, 1126, 3), np.uint8)
    hsv = cv2. cvtColor(img, cv2.COLOR_BGR2HSV)
    result = cv2.inRange(hsv, (110, .8*255, .9*255), (130, .95*255, 255))
    _, contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if cv2.contourArea(contour) > 85:
            cv2.drawContours(new_img, contour, -1, 255, 1)
            print cv2.contourArea(contour)

    cv2.circle(new_img, (game_state["PACMAN"][1], game_state["PACMAN"][0]), 25, (0, 255, 255))
    cv2.circle(new_img, (game_state["GHRED"][1], game_state["GHRED"][0]), 25, (0, 0, 255))
    cv2.circle(new_img, (game_state["GHBLUE"][1], game_state["GHBLUE"][0]), 25, (255, 255, 0))
    cv2.circle(new_img, (game_state["GHORANGE"][1], game_state["GHORANGE"][0]), 25, (75, 170, 233))
    cv2.circle(new_img, (game_state["GHPINK"][1], game_state["GHPINK"][0]), 25, (255, 185, 255))
    for pill in game_state["small_pills"]:
        cv2.circle(new_img, (pill[1], pill[0]), 7, (255, 255, 255))
    for pill in game_state["big_pills"]:
        cv2.circle(new_img, (pill[1], pill[0]), 25, (255, 255, 255))


    final = cv2.resize(new_img, (int(.5*1126), int(.5*1275)))
    #ret,gray = cv2.threshold(final, 0, (255, 0, 0), cv2.THRESH_BINARY)
    cv2.imshow("newtrack", final)
    cv2.waitKey(0)


def main():

    tic = time.clock()
    dictionary("screenshot_2.png")
    toc = time.clock()
    print("Processing Time: {}".format(toc-tic))

    #draw_track("screenshot_2.png")

    port = 1111
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:1111")

    pkled_data = pickle.dumps(game_state) 
    #pickle.dump(game_state, open("dict1.p", "wb"))

    while True:
        socket.send(pkled_data)
        time.sleep(1)

if __name__ == "__main__":
    main()
