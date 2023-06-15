import cv2
import numpy as np
from cv2 import contourArea

img1 = cv2.imread("images/1.jpg")
img2 = cv2.imread("images/2.jpg")
img3 = cv2.imread("images/3.jpg")

img1 = cv2.resize(img1, (1920, 1080))
img2 = cv2.resize(img2, (1920, 1080))
img3 = cv2.resize(img3, (1920, 1080))
img1 = img1[200:800, 600:900]
img2 = img2[200:800, 600:900]
img3 = img3[200:800, 600:900]

# print(img1.shape)


images = [img1, img2, img3]

# combine image
image = []
# image = cv2.hconcat(image,img3)


for count,i in enumerate(images):
    if(count == 0):
        image = i

    imgray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 130, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # biggest contour
    areas = [cv2.contourArea(c) for c in contours]
    filtercontour = []
    for c in contours:
        if contourArea(c) > 2:
            filtercontour.append(c)



    cv2.drawContours(i, filtercontour, -1, (0, 255, 0), 3)
    cv2.imshow("thresh",i)
    cv2.waitKey(0)
    if(count != 0):
        image = cv2.hconcat([image, i])

cv2.imshow("thresh",image)
cv2.waitKey(0)
