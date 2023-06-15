import cv2
import numpy as np

def measure_defocus(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

# Read an image file
blur = cv2.imread('/Users/badalchowdhary/PycharmProjects/DFD_ESI/venv/IMages/1.jpg')
focus = cv2.imread('/Users/badalchowdhary/PycharmProjects/DFD_ESI/venv/IMages/2.jpg')

defocusBlur = measure_defocus(blur)
defocusFocus = measure_defocus(focus)


print(defocusBlur)
print(defocusFocus)


# Convert the image to grayscale
# gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
# # Apply thresholding
# _, threshold = cv2.threshold(gray, 500, 255, cv2.THRESH_BINARY)
#
# # Find contours
# contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#
# Display the image
cv2.imshow('Image1', blur)
cv2.waitKey(0)
cv2.imshow('Image2', focus)
cv2.waitKey(0)

