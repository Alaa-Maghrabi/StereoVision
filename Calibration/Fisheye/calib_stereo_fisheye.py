import cv2 as cv
import os.path
import numpy as np
	
stereocalibration_flags = cv.fisheye.CALIB_USE_INTRINSIC_GUESS + cv.fisheye.CALIB_FIX_INTRINSIC + cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

cbrow = 7
cbcol = 5

data = np.load("fisheye_left_calibration.npz")
K_l = data['K']
D_l = data['D']


data = np.load("fisheye_right_calibration.npz")
K_r = data['K']
D_r = data['D']

for i in range(99,100):
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((1,cbrow*cbcol,3), np.float32)
	objp[0,:,:2]  = np.mgrid[0:cbcol,0:cbrow].T.reshape(-1,2)*45.5

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints_l = [] # 2d points in image plane.
	imgpoints_R = []
	print(i)
	for idx in range(0,i):
		imgL = cv.imread('Images_calibration/left_%d.png' %idx)
		imgR = cv.imread('Images_calibration/right_%d.png' %idx)
		
		# img=cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
		# img_shape = imgL.shape[:2]
		
		# map1, map2 = cv.fisheye.initUndistortRectifyMap(K_l, D_l, np.eye(3), K_l, img_shape[::-1], cv.CV_16SC2)
		# imgL = cv.remap(imgL, map1, map2, interpolation=cv.INTER_LINEAR,  borderMode=cv.BORDER_CONSTANT)
		
		# map1, map2 = cv.fisheye.initUndistortRectifyMap(K_r, D_r, np.eye(3), K_r, img_shape[::-1], cv.CV_16SC2)
		# imgR = cv.remap(imgR, map1, map2, interpolation=cv.INTER_LINEAR,  borderMode=cv.BORDER_CONSTANT)
		
		imgL_g=cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
		imgR_g=cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
		
		ret, corners_L= cv.findChessboardCorners(imgL_g, (cbrow,cbcol), None)
		ret1, corners_R = cv.findChessboardCorners(imgR_g, (cbrow,cbcol), None)

		if ret and ret1 and  (idx not in (10, 16,17, 34,38,37,23,52,54,86,98, 35, 25, 28,
										  29, 6, 42)):
			corners1 = cv.cornerSubPix(imgL_g,corners_L, (5,5), (-1,-1), criteria)
			corners2 = cv.cornerSubPix(imgR_g,corners_R, (5,5), (-1,-1), criteria)
			
			# cv.drawChessboardCorners(imgL, (cbrow,cbcol), corners1, ret)
			# cv.drawChessboardCorners(imgR, (cbrow,cbcol), corners2, ret)
			# cv.namedWindow('imgL %d' %idx)        # Create a named window
			# cv.moveWindow('imgL %d' %idx, 40,30)
			# cv.circle(imgL, (int(corners_L[0,0,0]), int(corners_L[0,0,1])), 10, (0, 255, 255), 1)
			# cv.imshow('imgL %d' %idx, imgL)
			# cv.namedWindow('imgR %d' %idx)        # Create a named window
			# cv.moveWindow('imgR %d' %idx, 800,30)
			# cv.circle(imgR, (int(corners_R[0, 0, 0]), int(corners_R[0, 0, 1])), 10, (0, 255, 255), 1)
			# cv.imshow('imgR %d' %idx, imgR)
			#
			# cv.waitKey(100)
			# cv.destroyAllWindows()
			
			objpoints.append(objp)
			imgpoints_l.append(corners_L)
			imgpoints_R.append(corners_R)


	R = np.zeros((3, 3), dtype=np.float64)
	T = np.zeros((3, 1), dtype=np.float64)
	N_OK = len(imgpoints_l)

	objpoints = np.array([objp]*len(imgpoints_l),dtype=np.float64)
	imgpoints_l = np.asarray(imgpoints_l, dtype=np.float64)
	imgpoints_R = np.asarray(imgpoints_R, dtype=np.float64)
	objpoints = np.reshape(objpoints, (N_OK, 1, cbcol*cbrow, 3))
	imgpoints_l = np.reshape(imgpoints_l, (N_OK, 1, cbcol*cbrow, 2))
	imgpoints_R = np.reshape(imgpoints_R, (N_OK, 1, cbcol*cbrow, 2))

	stereocalibration_retval, K_l, D_l, K_r, D_r, R, T = cv.fisheye.stereoCalibrate(objpoints,imgpoints_l,imgpoints_R,K_l, D_l, K_r, D_r,imgR_g.shape[::-1], R, T, criteria = criteria, flags = stereocalibration_flags)
	print(stereocalibration_retval)
	print('camera matrix \n', K_l)
	print('dist coeff \n', D_l)
	print('camera matrix 2\n', K_r)
	print('dist coeff 2\n', D_r)
	print('T \n', T)
	print('R \n', R)

	# K_l = np.array([[456.6911, 0, 336.1007],
	# 										[0, 457.1547, 242.7140],
	# 										[0, 0, 1]])
	#
	# K_r = np.array([[456.3592, 0, 322.7202],
	# 										[0, 457.1654, 259.4967],
	# 										[0, 0, 1]])

	
	#D_l = np.array([[-0.4146, 0.2693, -0.0009, -0.0020, -0.1496]])
	#D_r = np.array([[-0.3825, 0.1824, -0.0032, -0.00056, -0.0709]])
	
	T[0] = -71.6441
	T[1] = -1.5761
	T[2] = -0.9126
	
	R = np.array([[0.9994, 0.0048, 0.0342],
				[-0.0053, 0.9999, 0.0146],
				[-0.0341, -0.0148 ,0.9993]])

	R=np.transpose(R)

	R_l, R_r, P_l, P_r, Q = cv.fisheye.stereoRectify(K_l, D_l, K_r, D_r, imgR_g.shape[::-1], R, T, flags = 0, balance = 0 )
	print(Q)
	h,w = imgL.shape[:2]
	# undistort	
	map1, map2 = cv.fisheye.initUndistortRectifyMap(K_l, D_l, R_l, P_l, (w,h), cv.CV_32FC1)
	undistorted_img_l = cv.remap(imgL, map1, map2, interpolation=cv.INTER_LINEAR)


	map1, map2 = cv.fisheye.initUndistortRectifyMap(K_r, D_r, R_r, P_r, (w,h), cv.CV_32FC1)
	undistorted_img_r = cv.remap(imgR, map1, map2, interpolation=cv.INTER_LINEAR)
	for line in range(0, int(imgR.shape[0] / 20)):
		# img1[line * 20, :] = ((20-i)*5, i*2, i*10)
		# img2[line * 20, :] = ((20-i)*5, i*2, i*10)
		undistorted_img_l[line * 20, :] = ((20-i)*5, i*2, i*10)
		undistorted_img_r[line * 20, :] = ((20-i)*5, i*2, i*10)
		i = i + 1
	cv.imshow('image',np.hstack((imgL,imgR)))
	cv.imshow('image-after',np.hstack((undistorted_img_l,undistorted_img_r)))

	np.savez("fish_final_calib.npz", K1=K_l, D1=D_l, K2=K_r, D2=D_r,R1=R_l, R2=R_r, P1=P_l, P2=P_r, Q=Q)
	cv.waitKey()