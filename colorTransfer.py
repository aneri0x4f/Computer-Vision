"""
Color Transfer

Requirements:
- python
- basic knowledge on usage of comand line
- NumPy 

Algorithm:
- get a source, and target image
- convert both the source and the target image to the L*a*b* color space. 
- split source and target
- get the mean of target L*a*b channels y subtracting mean of each 
- normalize by getting its product with (standard div of source) and dividing by (standard div of target)
- add in the source mean
- clip any values that fall outside the range [0, 255]
- merge channels
- convert to RGB format

Recomendations:
- prefer a plain colored source image for it to act as a filter
"""

# import the necessary packages
import argparse

import cv2
import imutils
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="path to source image")
ap.add_argument("-t", "--target", required=True, help="path to source target")
args = vars(ap.parse_args())

# load image and convert to L*a*b
"""
Convert it to float32:
OpenCV represents images as multi-dimensional NumPy arrays, 
but defaults to the uint8 datatype. 
This is fine for most cases, but,
when performing the color transfer we could potentially 
have negative and decimal values, thus, 
we need to utilize the floating point data type.
"""
source = imutils.resize(cv2.imread(args["source"]), width = 700)
target = imutils.resize(cv2.imread(args["target"]), width = 700)

# split L*a*b and get stats for source and traget 

def image_stats(image):
	# return the mean and standard deviation of each channel
	(l, a, b) = cv2.split(image)

	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())

	return (lMean, lStd, aMean, aStd, bMean, bStd)

def color_transfer(source, target):
	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

	# make target mean = 0 
	(l, a, b) = cv2.split(target)
	l -= lMeanTar
	a -= aMeanTar
	b -= bMeanTar

	# normalize by the standard deviations
	l = (lStdSrc / lStdTar) * l 
	a = (aStdSrc / aStdTar) * a 
	b = (bStdSrc / bStdTar) * b 

	# add in the source mean
	l += lMeanSrc
	a += aMeanSrc
	b += bMeanSrc

	# clip the pixel intensities to [0, 255] 
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)

	# merge the channels together 
	# convert back to the RGB color
	# make sure to utilize the 8-bit unsigned integer data type
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

	# return the color transferred image
	return transfer    

cv2.imshow("Source", source)
cv2.imshow("Target", target)
cv2.imshow("Transfer", color_transfer(source, target))
cv2.waitKey(0)
