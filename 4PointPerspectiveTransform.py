"""
4 point perspective transform
Requirements:
- python
- basic knowledge on usage of comand line
- NumPy 
Approach:
- get the max of the contours
- get coordinates of the vertex from this countour
- order the coordinates (IMPORTANT)
- transform

Image source: 
- https://media-cdn.tripadvisor.com/media/photo-s/06/cf/0c/fe/our-bill.jpg
- http://clipart-library.com/images_k/sticky-note-transparent-background/sticky-note-transparent-background-21.png
"""

# import the necessary packages
import argparse

import cv2
import imutils
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to source image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = imutils.resize(image, width = 700)
cv2.imshow("image", image)

# preprocess and get max contour
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]


cs = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cs = imutils.grab_contours(cs)
c = max(cs, key = cv2.contourArea)

# get coordinate from the max chosen contour
def getCoordinates(image, c):

    vertices = []

    # approxPolyDP(): to perform an approximation of a shape of a contour.
    approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)

    # flatten the array containing the co-ordinates of the vertices.
    n = approx.ravel() 

    for i in range(len(n)) :
        if(i % 2 == 0):
            x = n[i]
            y = n[i + 1]
            vertices.append((x, y))

    return vertices            

pts = np.array(getCoordinates(image, c))

# get arranged coordinates
def order_points(pts):
    """
    We want a list(numpy array to be precise) of points with following arrangement:
    1st: top-left,
	2nd: top-right
    3rd: bottom-right
    4th: bottom-left 

    Logically for a certain perspective of the image, 
    top-left: min sum of coordinates
    bottom-right: max sum of coordinates

    top-right: min diff of coordinates
    bottom-left: max diff of coordinates
    """
    # allocate memory
    ordered = np.zeros((4, 2), dtype = "float32")

    # get sum, top-left, bottom-right 
    s = pts.sum(axis = 1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    # get diff, top-right, bottom-left
    diff = np.diff(pts, axis = 1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    return ordered

ordered = order_points(pts)

# transform
def four_point_transform(image, ordered):
    
	(tl, tr, br, bl) = ordered
	# compute the width of the new image
	bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	max_Width = max(int(bottom_width), int(top_width))

	# compute the height of the new image
	left_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	right_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	max_Height = max(int(left_height), int(right_height))

	# construct the set of destination points to obtain a "birds eye view"
    # order: bl, br, tr, tl
	dst = np.array([
		[0, 0],
		[max_Width - 1, 0],
		[max_Width - 1, max_Height - 1],
		[0, max_Height - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
    # parameters: numpy array of ordered coordinates, final coordinates 
	M = cv2.getPerspectiveTransform(ordered, dst)
	warped = cv2.warpPerspective(image, M, (max_Width, max_Height))

	# return the warped image
	return warped

cv2.imshow("birds eye view", four_point_transform(image, ordered))
cv2.waitKey(0)
