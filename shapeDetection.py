"""
SHAPE DETECTION and ANALYSIS:
you will be able to detect shapes from your image, 
and locate the centers on them.

Requirements:
- python
- basic knowledge on usage of comand line

Recommendation:
- prefer using images with dark backgrounds
- for other cases, you can update the threshold value to work with
"""

# import the necessary packages
import argparse

import cv2
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, blur it slightly, and threshold it
# Blurring to reduce high frequency noise to make our contour detection process more accurate.
# By thresholding, we are "binarizing the image" (black and white)
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# find and grab contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
	# compute the center of the contouredge case by adding a tiny number to denominator
    # avoid 
	M = cv2.moments(c)
	cX = int(M["m10"] / (M["m00"] + 1e-7))
	cY = int(M["m01"] / (M["m00"] + 1e-7))

	# draw the contour on the image
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    # mark the center of the shape on the image
    # parameters: image_name, coordinates, radius, color, thickness
    # use thickness of -1 px to fill the circle
	cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)

    # parameters: image_name, text, start_coordinates, font, font_scale, color, thickness 
	cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	
    # show the image
cv2.imshow("Image", image)
cv2.waitKey(0)
