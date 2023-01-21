"""
Finding extreme points in contours

Requirements:
- python
- basic knowledge on usage of comand line
- NumPy 
Recommendation:
- link to image: https://pyimagesearch.com/wp-content/uploads/2016/04/extreme_points_input.jpg
- prefer using images with dark backgrounds
- for other cases, you can update the threshold value to work with
"""

# import the necessary packages
import argparse

import cv2
import imutils
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# load and resize image
image = cv2.imread(args["image"])
image = imutils.resize(image,width = 600)

# conevert image to gray and blur it a bit
# blurring to reduce high frequency noise to make our contour detection process more accurate.

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# find and grab countours
# save the maxinum countour by area
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key = cv2.contourArea)

# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

# draw the outline of the object
cv2.drawContours(image, [c], -1, (0, 255, 255), 1)

# mark the extreme points with customized colors
cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
cv2.circle(image, extRight, 8, (0, 255, 0), -1)
cv2.circle(image, extTop, 8, (255, 0, 0), -1)
cv2.circle(image, extBot, 8, (255, 0, 255), -1)

cv2.imshow("Image", image)
cv2.waitKey(0)
