
# Python program for Detection of a  
# specific color(blue here) using OpenCV with Python 
import cv2 
import numpy as np  
  
# Webcamera no 0 is used to capture the frames 
cap = cv2.VideoCapture(1)
#cap1 = cv2.VideoCapture(1)
# cap.set(3, 800)
# cap.set(4, 480)
# cap1.set(3, 800)
# cap1.set(4, 480)

# This drives the program into an infinite loop. 
while(1):

	ret, frame_right = cap.read()
	#_, frame_left = cap1.read()


	cv2.imshow('frame_right',frame_right)
	#cv2.imshow('frame_left',frame_left)

 
	if cv2.waitKey(1) & 0xFF == ord('q'):
			break
  

cv2.destroyAllWindows() 
  

cap.release() 
cap1.release()
