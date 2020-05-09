import noise
import numpy as np
import cv2

img = cv2.imread('buried.jpg')
img = np.uint8(noise.noisy(img, 0.5))

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.desAllWindows()