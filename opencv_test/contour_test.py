import cv2
import numpy as np
  
# reading image
img = cv2.imread('Screenshot from 2022-10-25 13-25-23.png')

# converting image into grayscale image
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# setting threshold of gray image
#_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)
img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)
img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # erosion followed by dilation
img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) # dilation followed by erosion

bin_img = cv2.inRange(img_opening, (0, 0, 0), (0,0,0))
  
# using a findContours() function
contours, _ = cv2.findContours(
    bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours1, _ = cv2.findContours(
#     img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours2, _ = cv2.findContours(
#     img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours3, _ = cv2.findContours(
#     img_eroded_dialated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours4, _ = cv2.findContours(
#     img_dilated_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
i = 0
  
# list for storing names of shapes
for contour in contours:
  
    # here we are ignoring first counter because 
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue
  
    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)
      
    # using drawContours() function
    cv2.drawContours(img_opening, [contour], 0, (0, 0, 255), 5)
    #cv2.drawContours(img_erosion, [contour], 0, (0, 0, 255), 5)
    #cv2.drawContours(img_dilation, [contour], 0, (0, 0, 255), 5)
    #cv2.drawContours(img_opening, [contour], 0, (0, 0, 255), 5)
    #cv2.drawContours(img_dilated_eroded, [contour], 0, (0, 0, 255), 5)
  
    # finding center point of shape
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
  
    # putting shape name at center of each shape
    if len(approx) == 3:
        cv2.putText(img_opening, 'Triangle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  
    if len(approx) == 4:
        cv2.putText(img_opening, 'Quadrilateral', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  
    elif len(approx) == 5:
        cv2.putText(img_opening, 'Pentagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  
    elif len(approx) == 6:
        cv2.putText(img_opening, 'Hexagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  
    else:
        cv2.putText(img_opening, 'circle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  
# displaying the image after drawing contours
#cv2.imshow('img', img)
#cv2.imshow('erosion', img_erosion)
#cv2.imshow('dilation', img_dilation)
cv2.imshow('erosion-dilation', img_opening)
#cv2.imshow('dilation-erosion', img_dilated_eroded)

cv2.waitKey(0)
cv2.destroyAllWindows()