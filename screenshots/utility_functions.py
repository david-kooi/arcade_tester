
def extract_pacman(image_path):
    """
    @image_path: String of path to image file
    @returns: A tuple of pacman's (x,y) position
    """

    pass


def extract_blue_ghost(image_path):
    """
    @image_path: String of path to image file
    @returns: A tuple of blue ghost's (x,y) position
    """
    pass

def extract_pills(image_path):
    """
    @image_path: String of path to image file
    @returns: A list of tuples of pill (x,y) positions
              I.E: [(x1,y1),(x2,y2),...,(xn,yn)]
    """
    img = cv2.imread("screenshot_1.png")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    result = cv2.inRange(img, np.array([5, 28.8, 97.2]), np.array([9, 32.8, 101.2]))
    cv2. imshow(result, "pills")


