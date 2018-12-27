
# Python program for Detection of a  
# specific color(blue here) using OpenCV with Python 
import cv2 
import numpy as np  
import re
import subprocess

# device_re = re.compile("Bus\s+(?P<bus>\d+)\s+Device\s+(?P<device>\d+).+ID\s(?P<id>\w+:\w+)\s(?P<tag>.+)$", re.I)
# df = subprocess.check_output("lsusb").decode('utf-8')
# devices = []
# print(df)
# for i in df.split('\n'):
#     if i:
#         info = device_re.match(i)
#         if info:
#             dinfo = info.groupdict()
#             dinfo['device'] = '/dev/bus/usb/%s/%s' % (dinfo.pop('bus'), dinfo.pop('device'))
#             devices.append(dinfo)
# print(devices)

# Webcamera no 0 is used to capture the frames
cap = cv2.VideoCapture(1)
cap1 = cv2.VideoCapture(2)

cap.set(3, 640)
cap.set(4, 480)
cap1.set(3, 640)
cap1.set(4, 480)
cv2.waitKey(5000)



# This drives the program into an infinite loop.
while(1):
	ret, frame_right = cap.read()
	_, frame_left = cap1.read()

	cv2.imshow('frame_right',frame_right)
	cv2.imshow('frame_left',frame_left)


	if cv2.waitKey(30) & 0xFF == ord('q'):
			break


cv2.destroyAllWindows()


cap.release()
cap1.release()
