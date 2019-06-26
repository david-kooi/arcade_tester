import cv2
import numpy as np
img = cv2.imread(roi.jpg)
ball = img[280:340, 330:390]
img[260:320, 100:160] = ball
cv2.imshow("my_image".img)