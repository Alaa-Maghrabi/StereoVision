import cv2 as cv
assert cv.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'

import numpy as np
import os
import glob

CHECKERBOARD = (7,5)

subpix_criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv.fisheye.CALIB_CHECK_COND+cv.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for idx in range(0, 100):
	fname = 'Images_calibration/left_%d.png' %idx
	img = cv.imread(fname)
	if _img_shape == None:
		_img_shape = img.shape[:2]
	else:
		assert _img_shape == img.shape[:2], "All images must share the same size."
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
	ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
	if ret == True and idx not in [10,25,28,29,42,6]:
		objpoints.append(objp)
		cv.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
		imgpoints.append(corners)

N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

rms, _, _, _, _ = cv.fisheye.calibrate(objpoints,imgpoints,gray.shape[::-1],K,D,rvecs,tvecs,calibration_flags,(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
print(rms)
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

h,w = img.shape[:2]
dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
DIM = _img_shape[::-1]
dim2 = None
dim3 = None
balance = 0
assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"

if not dim2:
	dim2 = dim1

if not dim3:
	dim3 = dim1

scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
map1, map2 = cv.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv.CV_16SC2)
undistorted_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
np.savez("fisheye_left_calibration.npz", K=K, D=D)

for idx in range(0, 5):
	fname = 'left_%d.png' %idx
	img = cv.imread(fname)
	undistorted_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR,  borderMode=cv.BORDER_CONSTANT)
	cv.imshow("original_r", img)
	cv.imshow("undistorted_r", undistorted_img)

	cv.waitKey()
cv.destroyAllWindows()


##### RIGHT IMAGE

_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for idx in range(0, 100):
	fname = 'Images_calibration/right_%d.png' %idx
	img = cv.imread(fname)
	if _img_shape == None:
		_img_shape = img.shape[:2]
	else:
		assert _img_shape == img.shape[:2], "All images must share the same size."
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
	ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
	if ret == True and idx not in [10, 17, 34, 35]:
		objpoints.append(objp)
		cv.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
		imgpoints.append(corners)
		# cv.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
		# cv.namedWindow('imgL')        # Create a named window
		# cv.moveWindow('imgL', 40,30)
		# cv.imshow('imgL', img)
		
		# cv.waitKey()
		# cv.destroyAllWindows()


N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

rms, _, _, _, _ = cv.fisheye.calibrate(objpoints,imgpoints,gray.shape[::-1],K,D,rvecs,tvecs,calibration_flags,(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
print(rms)
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

np.savez("fisheye_right_calibration.npz", K=K, D=D)

h,w = img.shape[:2]
dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
DIM = _img_shape[::-1]
dim2 = None
dim3 = None
balance = 0
assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"

if not dim2:
	dim2 = dim1

if not dim3:
	dim3 = dim1

scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!

new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
map1, map2 = cv.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv.CV_16SC2)

for idx in range(0, 5):
	fname = 'right_%d.png' %idx
	img = cv.imread(fname) 
	undistorted_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR,  borderMode=cv.BORDER_CONSTANT)
	cv.imshow("original", img)
	cv.imshow("undistorted", undistorted_img)

	cv.waitKey()
cv.destroyAllWindows()