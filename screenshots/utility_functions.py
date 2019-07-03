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
    height = hsv.shape[0]
    width = hsv.shape[1]
    pill_px = np.where(result == 255)
    mid_first_pill = [0, 0]
    mid_last_pill = [0, 0]
    pill_dist = [0, 0]
    # get mid-point of firet and last pill
    for i in range(0, len(pill_px[1])):
        if pill_px[1][i+1] - pill_px[1][i] > 1:
            mid_first_pill[1] = (pill_px[1][i] + pill_px[1][0])/2
            pill_dist[1] = pill_px[1][i+1] - pill_px[1][0]
            break
    for i in range(1, len(pill_px[1])):
        if pill_px[1][len(pill_px[1])-i] - pill_px[1][len(pill_px[1])-i-1] > 1:
            mid_last_pill[1] = (pill_px[1][len(pill_px[1])-i] + pill_px[1][len(pill_px[1])-1])/2
            break
    for j in range(0, len(pill_px[0])):
        if pill_px[0][j+1] - pill_px[0][j] >= 2:
            mid_first_pill[0] = (pill_px[0][j] + pill_px[0][0])/2
            pill_dist[0] = pill_px[0][j+1] - pill_px[0][0]
            break
    for j in range(1, len(pill_px[0])):
        if pill_px[0][len(pill_px[0])-j] - pill_px[0][len(pill_px[0])-j-1] >= 2:
            mid_last_pill[0] = (pill_px[0][len(pill_px[0])-j] + pill_px[0][len(pill_px[0])-1])/2
            break
    pill_list = {"small_pills": [(0, 0)], "big_pills": [(0, 0)]}
    # check if pill is at intersection
    for j in range (0, 26):
        for i in range (0, 15):
            if np.any(result[mid_first_pill[0]+i*pill_dist[0], mid_first_pill[1]+j*pill_dist[1]] == 255):
                if i == 2 and (j == 0 or j == 25):
                    pill_list["big_pills"].append((mid_first_pill[0] + i * pill_dist[0], mid_first_pill[1] + j * pill_dist[1]))
                else:
                    pill_list["small_pills"].append((mid_first_pill[0]+i*pill_dist[0], mid_first_pill[1]+j*pill_dist[1]))
        for i in range (0, 14):
            if np.any(result[mid_last_pill[0]-i*pill_dist[0], mid_first_pill[1]+j*pill_dist[1]] == 255):
                if i == 6 and (j == 0 or j == 25):
                    pill_list["big_pills"].append((mid_last_pill[0] - i * pill_dist[0], mid_first_pill[1] + j * pill_dist[1]))
                else:
                    pill_list["small_pills"].append((mid_last_pill[0]-i*pill_dist[0], mid_first_pill[1]+j*pill_dist[1]))
    del pill_list["small_pills"][0]
    del pill_list["big_pills"][0]
    # draw circles at pill points
    for pill in pill_list["small_pills"]:
        cv2.circle(result, (pill[1], pill[0]), 10, 255)
    for pill in pill_list["big_pills"]:
        cv2.circle(result, (pill[1], pill[0]), 30, 255)
    # draw grid lines
    for i in range (0,26):
        cv2.line(result, (mid_first_pill[1]+i*pill_dist[1], 0), (mid_first_pill[1]+i*pill_dist[1], height), (255, 0, 0), 1)
    for i in range (0,15):
        cv2.line(result, (0, mid_first_pill[0]+i*pill_dist[0]), (width, mid_first_pill[0]+i*pill_dist[0]), (255, 0, 0), 1)
    for i in range (0,14):
        cv2.line(result, (0, mid_last_pill[0]-i*pill_dist[0]), (width, mid_last_pill[0]-i*pill_dist[0]), (255, 0, 0), 1)
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
    #extract_pacman("screenshot_1.png")
    find_pills("screenshot_1.png")
    draw_track("screenshot_1.png")

if __name__ == "__main__":
    main()


