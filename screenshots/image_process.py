import cv2
import numpy as np

img = cv2.imread("soccer.jpg")
ball = img[235:275, 275:320]
img[228:268, 43:88] = ball
img[226:266, 0:45] = ball
img[230:270, 129:174] = ball
cv2.imshow("my_image", img)
cv2.waitKey(0)