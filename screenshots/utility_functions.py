import cv2
import numpy as np
maxval = 1126

def circle_pacman(img):
    x_list, y_list = np.where(img == 255)
    avg_x = sum(x_list) / len(x_list)
    avg_y = sum(y_list) / len(y_list)
    cv2.circle(img, (avg_y, avg_x), 50, 255)

def resize_img(img):
    H = np.size(img, 0)
    W = np.size(img, 1)
    alpha = H/500
    downsize = cv2.resize(img, (int(W / alpha), int(H / alpha)))
    return downsize

def extract_pacman(image_path):
    """
    @image_path: String of path to image file
    @returns: A tuple of pacman's (x,y) position
    """
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask0 = cv2.inRange(img_hsv, (20, .8*255, .8*255), (40, 255, 255))
    downsize = resize_img(mask0)
    circle_pacman(downsize)
    cv2.imshow("pacman", downsize)
    zeros = np.zeros((1126, 1275), dtype=np.uint8)
    zeros[:5, :5] = 255

    indices = np.where(zeros == [255])
    print indices
    coordinates = zip(indices[0], indices[1])
    print coordinates
    cv2.waitKey(0)

def main():
    extract_pacman("screenshot_1.png")

if __name__ =="__main__":
    main()

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
    pass



