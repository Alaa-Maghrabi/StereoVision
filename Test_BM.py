import cv2 as cv
from numpy import inf
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import time
import datetime
from disparity_fisheye import Stereo
from scipy.signal import butter, lfilter, filtfilt

# def CallBackFunc(event, x, y, flags, param):
	# if event == cv.EVENT_LBUTTONDOWN:
		# print("Left button of the mouse is clicked - position (", x, ", ",y,",  RGB:", (100*fx * baseline) / (units * displ[y,x]) , ")")

def nothing(x):
  pass


# Filter requirements.
order = 3
fs = 30.0  # sample rate, Hz
cutoff = 2.  # desired cutoff frequency of the filter, Hz

def butter_lowpass():
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	return b, a


def butter_lowpass_filter(data):
	b, a = butter_lowpass()
	# y = lfilter(b, a, data)
	y = filtfilt(b, a, data, padlen=25)
	return y

cv.namedWindow('Colorbars')

# cv.namedWindow('Disparity Map')
# cv.setMouseCallback('Disparity Map', CallBackFunc)

window_size = 8
_minDisparity=0
a=8            # max_disp 
_blockSize=4
_disp12MaxDiff=50
_uniquenessRatio=3
_speckleWindowSize=5
_speckleRange=2
_preFilterCap=55

cv.createTrackbar("window_size", "Colorbars",7,255,nothing)
cv.createTrackbar("_minDisparity", "Colorbars",6,255,nothing)
cv.createTrackbar("a", "Colorbars",6,255,nothing)
cv.createTrackbar("_blockSize", "Colorbars",7,50,nothing)
cv.createTrackbar("_disp12MaxDiff", "Colorbars",30,250,nothing)
cv.createTrackbar("_uniquenessRatio", "Colorbars",1,50,nothing)
cv.createTrackbar("_speckleWindowSize", "Colorbars",5,20,nothing)
cv.createTrackbar("_speckleRange", "Colorbars",1,20,nothing)
cv.createTrackbar("_preFilterCap", "Colorbars",0,255,nothing)


cap = cv.VideoCapture(0)
cap1 = cv.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)
cap1.set(3, 640)
cap1.set(4, 480)
cv.waitKey(1000)
data = np.load("Parameters/fish_final_calib.npz")
K_l = data['K1']
K_r = data['K2']
D_l = data['D1']
D_r = data['D2']
R_l = data['R1']
R_r = data['R2']
P_l = data['P1']
P_r = data['P2']
Q = data['Q']
print(Q)
print(K_l)
print(K_r)
print(D_l)
print(D_r)
cbrow =  7
cbcol = 5
# Number of frames to capture
num_frames = 900;
 
fx = 404.22        # lense focal length
baseline = 1/0.01264   # distance in mm between the two cameras
units = 1  # depth units, adjusted for the output to fit in one byte

#print ("Capturing {0} frames".format(num_frames))

h = 480
w = 640
map1l, map2l = cv.fisheye.initUndistortRectifyMap(K_l, D_l, R_l, P_l, (w,h), cv.CV_32FC1)
#map1l, map2l = cv.initUndistortRectifyMap(K_l, D_l, R_l, P_l, (w,h), cv.CV_32FC1)
map1r, map2r = cv.fisheye.initUndistortRectifyMap(K_r, D_r, R_r, P_r, (w,h), cv.CV_32FC1)
#map1r, map2r = cv.initUndistortRectifyMap(K_r, D_r, R_r, P_r, (w,h), cv.CV_32FC1)

left_matcher = cv.StereoSGBM_create(
	minDisparity=-_minDisparity,
	numDisparities=16*a,             # max_disp has to be dividable by 16 f. E. HH 192, 256
	blockSize=_blockSize,
	P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
	P2=32 * 3 * window_size ** 2,
	disp12MaxDiff=_disp12MaxDiff,
	uniquenessRatio=_uniquenessRatio,
	speckleWindowSize=_speckleWindowSize,
	speckleRange=_speckleRange,
	preFilterCap=_preFilterCap,
	mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)

right_matcher = cv.ximgproc.createRightMatcher(left_matcher)


# FILTER Parameters
lmbda = 80000
sigma = 1.3
visual_multiplier = 1.0

wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

# ret, frame = cap1.read()
# ret1, frame1 = cap.read()

# imgL = cv.remap(frame1, map1l, map2l, interpolation=cv.INTER_LINEAR)
# imgR = cv.remap(frame, map1r, map2r, interpolation=cv.INTER_LINEAR)
# imgL=cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
# imgR=cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
# displ = left_matcher.compute(imgL, imgR).astype(np.float32)/16
# dispr = right_matcher.compute(imgR, imgL).astype(np.float32)/16
# Start time
start = time.time()
#f = open("test_example.txt", "w+")
graph = [] 
#while(True) :
for frames in range(num_frames):
    # Capture frame-by-frame
	ret, frame = cap1.read()
	ret1, frame1 = cap.read()
	#frame = cv.imread('images/left_%d.png' %idx)
	#frame1 = cv.imread('images/right_%d.png' %idx)


	imgL = cv.remap(frame1, map1l, map2l, interpolation=cv.INTER_LINEAR)
	imgR = cv.remap(frame, map1r, map2r, interpolation=cv.INTER_LINEAR)
	
	i= 0
	imageleft=imgL.copy()
	imageright=imgR.copy()
	
	for line in range(0, int(imgR.shape[0] / 20)):
			imageleft[line * 20, :] = ((20-i)*5, i*5, i*10)
			imageright[line * 20, :] = ((20-i)*5, i*5, i*10)
			i = i + 1

	imgL=cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
	imgR=cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
	
	window_size=cv.getTrackbarPos("window_size", "Colorbars")
	_minDisparity=cv.getTrackbarPos("_minDisparity", "Colorbars")
	a=cv.getTrackbarPos("a", "Colorbars")
	_blockSize=cv.getTrackbarPos("_blockSize", "Colorbars")
	_disp12MaxDiff=cv.getTrackbarPos("_disp12MaxDiff", "Colorbars")
	_uniquenessRatio=cv.getTrackbarPos("_uniquenessRatio", "Colorbars")
	_speckleWindowSize=cv.getTrackbarPos("_speckleWindowSize", "Colorbars")
	_speckleRange=cv.getTrackbarPos("_speckleRange", "Colorbars")
	_preFilterCap=cv.getTrackbarPos("_preFilterCap", "Colorbars")
	
	left_matcher = cv.StereoSGBM_create(
	minDisparity=-_minDisparity,
	numDisparities=16*a,             # max_disp has to be dividable by 16 f. E. HH 192, 256
	blockSize=_blockSize,
	P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
	P2=32 * 3 * window_size ** 2,
	disp12MaxDiff=_disp12MaxDiff,
	uniquenessRatio=_uniquenessRatio,
	speckleWindowSize=_speckleWindowSize,
	speckleRange=_speckleRange,
	preFilterCap=_preFilterCap,
	mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
	
	displ = left_matcher.compute(imgL, imgR).astype(np.float32)/16
	#dispr = right_matcher.compute(imgR, imgL).astype(np.float32)/16

	displ = np.int16(displ)
	#dispr = np.int16(dispr)
	#filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
	#norm_image_r = cv.normalize(dispr, None, alpha = 0, beta = 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
	norm_image_l = cv.normalize(displ, None, alpha = 0, beta = 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
	#filteredImg_ = cv.normalize(src=filteredImg, dst=filteredImg, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F);
	#filteredImg = np.uint8(filteredImg)
	#filteredImg = filteredImg_
	#cv.imshow('image_',displ)
	#image_3d = cv.reprojectImageTo3D(displ, Q)
	
	hsv = cv.cvtColor(imageleft, cv.COLOR_BGR2HSV)
	mask = cv.inRange(hsv, (0, 100, 20), (20, 255, 255))
	mask = cv.erode(mask,  None, iterations=1)
	mask = cv.dilate(mask, None, iterations=2)
	#cv.imshow('mask', mask)
	# # Setup SimpleBlobDetector parameters.
	# params = cv.SimpleBlobDetector_Params()
	 
	# # Change thresholds
	# params.minThreshold = 200
	# params.maxThreshold = 500
	# params.minDistBetweenBlobs= 50
	
	# # Filter by Area.
	# params.filterByArea = True
	# params.minArea = 1500
	 
	# params.filterByColor=True
	# params.blobColor=255
	 
	# # Filter by Convexity
	# params.filterByConvexity = False
	# params.minConvexity = 0.87
	 
	# # Filter by Inertia
	# params.filterByInertia = False
	# params.minInertiaRatio = 0.01

	# # Set up the detector with default parameters.
	# detector = cv.SimpleBlobDetector_create(params)
	 
	# # Detect blobs.
	# keypoints = detector.detect(mask)
	 
	# # Draw detected blobs as red circles.
	# im_with_keypoints = cv.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			
	# gray_image = cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2GRAY)
	# # convert the grayscale image to binary image
	# ret,thresh = cv.threshold(gray_image,127,255,0)
	
	# cv.imshow('im', im_with_keypoints)
	# # calculate moments of binary image
	# M = cv.moments(thresh)
	cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
							cv.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	x, xc = -999, -999
	y, yc = -999, -999
	radius = -999

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key= cv.contourArea)
		((x, y), radius) = cv.minEnclosingCircle(c)
		# computing the centroid of the ball
		M = cv.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv.circle(imageleft, (int(x), int(y)), int(radius), (0, 255, 255), 1)
			cv.circle(imageleft, center,           2, (0, 255, 255),  -1)

			xc = int(x)
			yc = int(y)
			R = np.array([[xc],
							[yc],
							[displ[yc,xc]],
							[1]])
			point_3d = Q.dot(R)
			image_3d = point_3d[0:3]/point_3d[3]
			#print(image_3d[2,0], (fx * baseline) / (units * displ[yc,xc]))
			if(image_3d[2] < 2000 and image_3d[2] > 100):
				t = time.time()
				seconds = t - start
				print(image_3d)
				graph.append([seconds, image_3d[0,0],image_3d[1,0],image_3d[2,0]])
				#f.write("{}, {}, {}, {} \n".format(seconds, image_3d[0,0],image_3d[1,0],image_3d[2,0]))
				#cv.putText(imageleft,"{0:.2f}".format(image_3d[0,0]), (xc+40,yc+0), cv.FONT_HERSHEY_SIMPLEX, 0.75, 255)
				#cv.putText(imageleft,"{0:.2f}".format(image_3d[1,0]), (xc+40,yc+20), cv.FONT_HERSHEY_SIMPLEX, 0.75, 255)
				#cv.putText(imageleft,"{0:.2f}".format(image_3d[2,0]), (xc+40,yc+40), cv.FONT_HERSHEY_SIMPLEX, 0.75, 255)
				#graph_rectified.append((fx * baseline) / (units * filteredImg[cY,cX]))


	# show the frame to our screen
	#cv.imshow("tracking_ball", imageleft)
	
	# calculate x,y coordinate of center
	# if(M["m00"]==0):
		# cX = 1
		# cY = 1
	# else:
		# cX = int(M["m10"] / M["m00"])
		# cY = int(M["m01"] / M["m00"])
		# #print(image_3d[cY,cX,:])
		# R = np.array([[cX],
						# [cY],
						# [displ[cY,cX]],
						# [1]])
		# point_3d = Q.dot(R)
		# image_3d = point_3d[0:3]/point_3d[3]
		# #print(image_3d[2,0])
		# if(image_3d[2] < 2000 and image_3d[2] > 100):
			# t = time.time()
			# seconds = t - start
			# print(image_3d)
			# graph.append([seconds, image_3d[0,0],image_3d[1,0],image_3d[2,0]])
			#f.write("{}, {}, {}, {} \n".format(seconds, image_3d[0,0],image_3d[1,0],image_3d[2,0]))
			# graph_rectified.append((fx * baseline) / (units * filteredImg[cY,cX]))
		
	#cv.circle(norm_image_r,(cX,cY),2,(0,255,255),3)
	cv.circle(norm_image_l,(xc,yc),2,(0,255,255),3)
	#cv.circle(imageright,(cX,cY),2,(0,255,255),3)
	#cv.circle(imageleft,(cX,cY),2,(0,255,255),3)
	#cv.circle(displ,(xc,yc),2,(0,255,255),3)
	#cv.imshow('colors', displ)
	#cv.imshow('disparity_r', norm_image_r)
	cv.imshow('disparity_l', norm_image_l)
	cv.imshow('image',np.hstack((imageright,imageleft)))
	
	
	
	if cv.waitKey(1) & 0xFF == ord('q'):
		break


# End time
end = time.time()

# Time elapsed
seconds = end - start
print ("Time taken : {0} seconds".format(seconds))

# Calculate frames per second
fps  = num_frames / seconds;
print ("Estimated frames per second : {0}".format(fps))

graph = np.array(graph)

# When everything done, release the capture
cap.release()
cap1.release()
cv.destroyAllWindows()
#f.close()
plt.figure('x cm')
plt.plot(graph[:,1], label='3D')
y1 = butter_lowpass_filter(graph[:, 1])
plt.plot(y1, label='3D fil')
plt.legend(loc='upper left')
plt.figure('y cm')
plt.plot(graph[:,2], label='3D')
y2 = butter_lowpass_filter(graph[:, 2])
plt.plot(y2, label='3D fil')
plt.legend(loc='upper left')
plt.figure('z cm')
plt.plot(graph[:,3], label='3D')
y3 = butter_lowpass_filter(graph[:, 3])
plt.plot(y3, label='3D fil')
plt.legend(loc='upper left')
plt.show()