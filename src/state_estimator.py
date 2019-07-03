import cv2
import numpy as np


# Sprite colors in HSV
PAC_YELLOW = (60, 255, 255)
GH_RED = (0.25, 255, .949*255)
GH_BLUE = (180.24, .988*255, .9765*255)
GH_ORANGE = (32.69, .5847*255, .9725*255)
GH_PINK = (301.85, .261*255, .9765*255)
PILL_RED = (9, .3175*255, .9882*255)
BORDER_BLUE = (241.52, .7453*255, .4157*255)
COLOR_LIST = [PAC_YELLOW, GH_RED, GH_BLUE, GH_ORANGE, GH_PINK]

def extract_COLORS(image_path):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for color in COLOR_LIST:
        lower_bound = ((color[0]-30)/2, color[1]-30, color[2]-30)
        upper_bound = ((color[0]+30)/2, color[1]+30, color[2]+30)
        mask0 = cv2.inRange(img_hsv, lower_bound, upper_bound)
        cv2.imshow("sprites", mask0)
        x_output, y_output = get_avg_pos(mask0)
        print x_output, y_output
        cv2.waitKey(0)


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

def process_pills(PILL_RED, hsv_img):
    """
    @input hsv_img: HSV type image
    @input PILL_RED: isolate red pills
    @returns: xy values of red pills
    """
    pass
def process_obs(BORDER_BLUE, hsv_img):
    """
    @input hsv_img: HSV type image
    @input BORDER_BLUE: isolate blue border 
    @returns: processes locations of blue border
    """
    pass
def main():
    extract_COLORS("screenshot_2.png")

if __name__ == "__main__":
    main()




