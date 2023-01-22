"""
Get co-ordinates of vertices from the contour
Requirements:
- python
- basic knowledge on usage of comand line
- NumPy 
Approach:
-The co-ordinates of each vertices of a contour is hidden in the contour itself. 
- We will be using numpy library to convert all the co-ordinates of a contour 
into a linear array. This linear array would contain the x and y co-ordinates 
of each contour. 
- The key point here is that the first co-ordinate in the array would always 
be the co-ordinate of the topmost vertex and hence could help in detection of 
orientation of an image.
Recomendations:
- prefer an image with dark background
- for other cases you can handle the threshold or use inRange instead
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

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]


cs = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cs = imutils.grab_contours(cs)

def getCoordinates(image, cs):

    vertices = []
    for c in cs :

        # approxPolyDP(): to perform an approximation of a shape of a contour.
        approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
    
        # draws boundary of contours
        cv2.drawContours(image, [approx], 0, (0, 0, 255), 5) 
    
        # flatten the array containing the co-ordinates of the vertices.
        n = approx.ravel() 
    
        for i in range(len(n)) :
            if(i % 2 == 0):
                x = n[i]
                y = n[i + 1]
                vertices.append([x, y])
                
                # OPTIONAL:
                # printing coordinates
                # String containing the co-ordinates.
                coord = str(x) + " " + str(y) 
        
                if(i == 0):
                    # text on topmost co-ordinate.
                    cv2.putText(image, "Arrow tip", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0)) 
                else:
                    # text on remaining co-ordinates.
                    cv2.putText(image, coord, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0)) 

    return vertices            

print(getCoordinates(image, cs))           
cv2.imshow("image", image)

cv2.waitKey(0)
