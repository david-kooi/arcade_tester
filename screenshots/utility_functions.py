import cv2
import numpy as np

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
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    result = cv2.inRange(hsv, (0, .2*255, .9*255), (20, .4*255, 255))
    final = cv2.resize(result, (int(0.5*1126), int(0.5*1275)))
    cv2. imshow("pills", final)
    cv2.waitKey(0)

def find_pills(image_path):
    img = cv2. imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    result = cv2.inRange(hsv, (0, .2*255, .9*255), (20, .4*255, 255))
    cv2.imshow("before", result)
    height = hsv.shape[0]
    width = hsv.shape[1]
    pill_px = np.where(result == 255)
    mid_first_pill = [0, 0]
    pill_dist = [0, 0]
    for i in range(0, len(pill_px[1])):
        if pill_px[1][i+1] - pill_px[1][i] > 1:
            mid_first_pill[1] = (pill_px[1][i] + pill_px[1][0])/2
            pill_dist[1] = pill_px[1][i+1] - pill_px[1][0]
            break
    for j in range(0,len(pill_px[0])):
        if pill_px[0][j+1] - pill_px[0][j] >= 2:
            mid_first_pill[0] = (pill_px[0][j] + pill_px[0][0])/2
            pill_dist[0] = pill_px[0][j+1] - pill_px[0][0]
            break
    pill_list = {"small_pills": [(0, 0)], "big_pills": [(0, 0)]}
    for i in range (0, 29):
        for j in range (0, 26):
            if np.any(img[mid_first_pill[0]+i*pill_dist[0], mid_first_pill[1]+j*pill_dist[1]] == 255):
                pill_list["small_pills"].append((mid_first_pill[0]+i*pill_dist[0], mid_first_pill[1]+j*pill_dist[1]))
    for pill in pill_list["small_pills"]:
        cv2.circle(result, (pill[1], pill[0]), 7, 255)
    pill_list["small_pills"].remove((0, 0))
    for i in range (0,26):
        cv2.line(result, (mid_first_pill[1]+i*pill_dist[1], 0), (mid_first_pill[1]+i*pill_dist[1], height), (255, 0, 0), 1)
    for i in range (0,29):
        cv2.line(result, (0, mid_first_pill[0]+i*pill_dist[0]), (width, mid_first_pill[0]+i*pill_dist[0]), (255, 0, 0), 1)
    final = cv2.resize(result, (int(.5*1126), int(.5*1275)))
    ret,gray = cv2. threshold(final, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow("after", gray)
    cv2.waitKey(0)

def draw_track(image_path):
    img = cv2. imread(image_path)
    new_img = np.zeros((1275, 1126, 1), np.uint8)
    hsv = cv2. cvtColor(img, cv2.COLOR_BGR2HSV)
    result = cv2.inRange(hsv, (110, .8*255, .9*255), (130, .95*255, 255))
    cv2.imshow("track", result)
    cv2.waitKey(0)
    _, contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if cv2.contourArea(contour) > 85:
            cv2.drawContours(new_img, contour, -1, 255, 1)
            print cv2.contourArea(contour)
    final = cv2.resize(new_img, (int(.5*1126), int(.5*1275)))
    ret,gray = cv2. threshold(final, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow("newtrack", gray)
    cv2.waitKey(0)



def main():
    '''extract_pills("screenshot_1.png")'''
    '''find_pills("screenshot_1.png")'''
    draw_track("screenshot_1.png")

if __name__ == "__main__":
    main()