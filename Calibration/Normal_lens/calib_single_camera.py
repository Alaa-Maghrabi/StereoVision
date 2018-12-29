import numpy as np
import cv2 as cv
import glob	

cbrow = 7
cbcol = 5

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow*cbcol,3), np.float32)
objp[:,:2] = np.mgrid[0:cbcol,0:cbrow].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for idx in range(0, 19):
	fname = 'calibration_images/left_%d.png' %idx
	img = cv.imread(fname)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
	ret, corners = cv.findChessboardCorners(gray, (cbcol,cbrow), None)
	
	# Remove the 5th picture for left and the 14th for right
	# if (idx == 5):
		# ret = 0
	
    # If found, add object points, image points (after refining them)
	if ret == True:
		#print(idx)
		objpoints.append(objp)
		corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
		imgpoints.append(corners)
		#Draw and display the corners
		# cv.drawChessboardCorners(img, (cbcol,cbrow), corners2, ret)
		# cv.imshow('img', img)
		# cv.waitKey()
		
#print(objpoints)
#print(imgpoints)

ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#np.savez("Calibration/normal_left_calibration.npz", ret=ret_l, mtx=mtx_l, dist=dist_l, rvecs=rvecs_l, tvecs=tvecs_l)

# Check for left image
img = cv.imread('calibration_images/left_2.png')
h,  w = img.shape[:2]
newcameramtx_l, roi_l = cv.getOptimalNewCameraMatrix(mtx_l, dist_l, (w,h), 1, (w,h))
print(mtx_l, '\n', newcameramtx_l, roi_l)

# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx_l, dist_l, None, newcameramtx_l, (w,h), 5)
dst_l = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi_l
#dst_l = dst_l[y:y+h, x:x+w]
cv.imshow('before calib', img)
cv.imshow('calibresult', dst_l)
cv.waitKey()
tot_error = 0
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs_l[i], tvecs_l[i], mtx_l, dist_l)
    error = cv.norm(imgpoints[i],imgpoints2, cv.NORM_L2)/len(imgpoints2)
    tot_error += error

print ("total error: ", mean_error/len(objpoints))

# Second Camera

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for idx in range(0, 19):
	fname = 'right_%d.png' %idx
	img = cv.imread(fname)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
	ret, corners = cv.findChessboardCorners(gray, (cbcol,cbrow), None)
    # If found, add object points, image points (after refining them)
	# if (idx == 14):
		# ret = 0
		
	if ret == True:
		objpoints.append(objp)
		corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
		imgpoints.append(corners)
		# Draw and display the corners
		# cv.drawChessboardCorners(img, (cbcol,cbrow), corners2, ret)
		# cv.imshow('img', img)
		# cv.waitKey()

ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#np.savez("Calibration/normal_right_calibration.npz", ret=ret_r, mtx=mtx_r, dist=dist_r, rvecs=rvecs_r, tvecs=tvecs_r)


# Check for right image
img = cv.imread('right_15.png')
h,  w = img.shape[:2]
newcameramtx_r, roi_r = cv.getOptimalNewCameraMatrix(mtx_r, dist_r, (w,h), 1, (w,h))
print(newcameramtx_r, roi_r)

# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx_r, dist_r, None, newcameramtx_r, (w,h), 5)
dst_r = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi_r
#dst_r = dst_r[y:y+h, x:x+w]
cv.imshow('before calib_r', img)
cv.imshow('calibresult_r', dst_r)

tot_error = 0
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs_r[i], tvecs_r[i], mtx_r, dist_r)
    error = cv.norm(imgpoints[i],imgpoints2, cv.NORM_L2)/len(imgpoints2)
    tot_error += error

print ("total error: ", mean_error/len(objpoints))

cv.waitKey()
cv.destroyAllWindows()
