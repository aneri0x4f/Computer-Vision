"""
Sort contours
Aim:
Given an image, we will detect the contours, and enclose the objects in a box.
these can be sorted and ranked based on their location: top-bottom, bottom-top, left-right and right-left.
Requirements:
- python
- basic knowledge on usage of comand line
- NumPy 
- cv2, imutils libraries 
Approach:
- get variables to inform teh sorting as per Y axis(top-bottom/ bottom-top) or X axis(left-right/ right-left).
- other var contains information about the reversed sorting = True or False (bottom-top/ right-left)
- we sort using zip and sorted functions, with parameters to sorted as reverse(bool), axis(binary)
- we also have a display method that adds text to the image contours in order
"""

import numpy as np
import argparse
import imutils
import cv2

# function to sort 
# parameters: conotours in the image, method to sort(left-to-right, right-to-left, top-to-bottom, bottom-to-top)
# returns sorted contours and bounding box
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
  
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
    
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
    
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
  
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

# adds ranked text on the image 
# parameters: image, specific contour, rank
# returns image
def draw_contour(image, c, i):
	# compute the center of the contour area and draw a circle
	# representing the center
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
  
	# draw the countour number on the image
  # parameters: image, text, initial point, font, font scale, color, thickness
	cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
	# return the image with the contour number drawn on it
	return image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-m", "--method", required=True, help="Sorting method")
args = vars(ap.parse_args())

# load the image and initialize the accumulated edge image
image = cv2.imread(args["image"])
# allocate memory for the edge map on Line 52.
accumEdged = np.zeros(image.shape[:2], dtype="uint8")

# loop over the blue, green, and red channels, respectively
for chan in cv2.split(image):
	# blur the channel, extract edges from it, and accumulate the set of edges for the image
	chan = cv2.medianBlur(chan, 11)
	edged = cv2.Canny(chan, 50, 200)
	accumEdged = cv2.bitwise_or(accumEdged, edged)
  
# show the accumulated edge map
cv2.imshow("Edge Map", accumEdged)

# find contours in the accumulated image, keeping only the largest ones
cnts = cv2.findContours(accumEdged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
orig = image.copy()
# loop over the (unsorted) contours and draw them
for (i, c) in enumerate(cnts):
	orig = draw_contour(orig, c, i)
  
# show the original, unsorted contour image
cv2.imshow("Unsorted", orig)

# sort the contours according to the provided method
(cnts, boundingBoxes) = sort_contours(cnts, method=args["method"])

# loop over the (now sorted) contours and draw them
for (i, c) in enumerate(cnts):
	draw_contour(image, c, i)
  
# show the output image
cv2.imshow("Sorted", image)
cv2.waitKey(0)
