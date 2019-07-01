import os 

ROOT = os.environ['ARCADE_FR_ROOT']


# Sprite colors in HSV
PAC_YELLOW   = (1,1,1) 
GHOST_RED    = (1,1,1)
GHOST_BLUE   = (1,1,1)
GHOST_ORANGE = (1,1,1)
GHOST_PINK   = (1,1,1)
PILL_RED     = (1,1,1)
BORDER_BLUE  = (1,1,1)
COLOR_LIST   = [PAC_YELLOW, GH_RED, GH_BLUE, GH_ORANGE, GH_PINK]

def get_img(img_path):
    """
    @input img_path: Full path of the image
    @returns: A BGR image of the image path 
    """

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



if __name__ == "__main__":
    hsv_img = get_img()

    sprite_xy = process_sprite(COLOR_LIST, hsv_img) 
    pill_xy   = process_pills(PILL_RED, hsv_img)
    obs_hulls = process_obs(BORDER_BLUE, hsv_img)





