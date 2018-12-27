import cv2 as cv
import os.path
import numpy as np

	
stereocalibration_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-6)
stereocalibration_flags = cv.CALIB_USE_INTRINSIC_GUESS
#stereocalibration_flags =  (cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_ZERO_TANGENT_DIST +cv.CALIB_SAME_FOCAL_LENGTH)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cbrow = 7
cbcol = 5


data = np.load("../../Parameters/normal_left_calibration.npz")
ret_l = data['ret']
mtx_l = data['mtx']
dist_l = data['dist']
rvecs_l = data['rvecs']
tvecs_l = data['tvecs']

data = np.load("../../Parameters/normal_right_calibration.npz")
ret_r = data['ret']
mtx_r = data['mtx']
dist_r = data['dist']
rvecs_r = data['rvecs']
tvecs_r = data['tvecs']


for i in range(19,20):
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((cbrow*cbcol,3), np.float32)
	objp[:,:2] = np.mgrid[0:cbcol,0:cbrow].T.reshape(-1,2)*45.5

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints_l = [] # 2d points in image plane.
	imgpoints_R = []


	for idx in range(0, i):
		imgL = cv.imread('calibration_images/left_%d.png' %idx)
		imgR = cv.imread('calibration_images/right_%d.png' %idx)
		
		h,  w = imgL.shape[:2]

		# map1, map2 = cv.fisheye.initUndistortRectifyMap(K_l, D_l, np.eye(3), K_l, (w,h), cv.CV_32FC1)
		# imgL = cv.remap(imgL, map1, map2, interpolation=cv.INTER_LINEAR)

		# map1, map2 = cv.fisheye.initUndistortRectifyMap(K_r, D_r, np.eye(3), K_r, (w,h), cv.CV_32FC1)
		# imgR = cv.remap(imgR, map1, map2, interpolation=cv.INTER_LINEAR)
		
		imgL_g=cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
		imgR_g=cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
		
		ret, corners_L= cv.findChessboardCorners(imgL_g, (cbrow,cbcol), None)
		ret1, corners_R = cv.findChessboardCorners(imgR_g, (cbrow,cbcol), None)
		
		if ret and ret1 and  (idx not in ()):
			corners1 = cv.cornerSubPix(imgL_g,corners_L, (5,5), (-1,-1), criteria)
			corners2 = cv.cornerSubPix(imgR_g,corners_R, (5,5), (-1,-1), criteria)
			# Draw and display the corners
			# cv.drawChessboardCorners(imgL, (cbrow,cbcol), corners1, ret)
			# cv.drawChessboardCorners(imgR, (cbrow,cbcol), corners2, ret)
			# cv.namedWindow('imgL %d' %idx)        # Create a named window
			# cv.moveWindow('imgL %d' %idx, 40,30)
			# cv.imshow('imgL %d' %idx, imgL)
			# cv.namedWindow('imgR %d' %idx)        # Create a named window
			# cv.moveWindow('imgR %d' %idx, 800,30)
			# cv.imshow('imgR %d' %idx, imgR)
			
			# cv.waitKey()
			# cv.destroyAllWindows()
			objpoints.append(objp)
			imgpoints_l.append(corners_L)
			imgpoints_R.append(corners_R)

	
	stereocalibration_retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv.stereoCalibrate(objpoints,imgpoints_l,imgpoints_R,mtx_l,dist_l,mtx_r,dist_r,imgR_g.shape[::-1], criteria = stereocalibration_criteria, flags = stereocalibration_flags)

	print(stereocalibration_retval)
	print('camera matrix \n', cameraMatrix1)
	print('dist coeff \n', distCoeffs1)
	print('camera matrix 2\n', cameraMatrix2)
	print('dist coeff 2\n', distCoeffs2)
	print('T \n', T)
	print('R \n', R)
	cameraMatrix2 = np.array([[631.8552, 0, 320.2124],
											[0, 632.6490, 233.4695],
											[0, 0, 1]])
	
	cameraMatrix1 = np.array([[638.7672, 0, 310.1864],
											[0, 636.8511, 242.3637],
											[0, 0, 1]])
	
	
	distCoeffs2 = np.array([[0.063, -0.0562, 0.0016, 0.0006534, -0.5001]])
	distCoeffs1 = np.array([[0.057, 0.0763, 0.0026, -0.00094, -1.0336]])
	T[0] = 74.5035
	T[1] = -0.1796
	T[2] = -2.4537
	
	R = np.array([[0.9996, 0.0146,0.0272],
				[-0.0145, 0.9999, -0.0042],
				[-0.0272, 0.0038 ,0.9996]])
	R=np.transpose(R)
	print('camera matrix \n', cameraMatrix1)
	print('dist coeff \n', distCoeffs1)
	print('camera matrix 2\n', cameraMatrix2)
	print('dist coeff 2\n', distCoeffs2)
	print('T \n', T)
	print('R \n', R)
	
	(rect_l, rect_r,proj_l, proj_r, Q, roi_l, roi_r)  = cv.stereoRectify(cameraMatrix1, distCoeffs1,cameraMatrix2,distCoeffs2,imgR_g.shape[::-1],R,T,flags=1,alpha=0)

	print(proj_l)
	print(proj_r)

	mtx_l = cameraMatrix1
	dist_l = distCoeffs1
	mtx_r = cameraMatrix2
	dist_r = distCoeffs2

	h,  w = imgL.shape[:2]
	# undistort	
	mapx, mapy = cv.initUndistortRectifyMap(mtx_l, dist_l, rect_l, proj_l, (w,h), cv.CV_32FC1)
	dst_l = cv.remap(imgL, mapx, mapy, cv.INTER_LINEAR)

	# undistort	
	mapx, mapy = cv.initUndistortRectifyMap(mtx_r, dist_r, rect_r, proj_r, (w,h), cv.CV_32FC1)
	dst_r = cv.remap(imgR, mapx, mapy, cv.INTER_LINEAR)

	# check calibration
	undistorted_l = cv.undistortPoints(np.concatenate(imgpoints_l).reshape(-1,1, 2), mtx_l, dist_l, P=mtx_l)
	lines_l = cv.computeCorrespondEpilines(undistorted_l, 1, F)
	undistorted_r = cv.undistortPoints(np.concatenate(imgpoints_R).reshape(-1,1, 2), mtx_r, dist_r, P=mtx_r)
	lines_r = cv.computeCorrespondEpilines(undistorted_r, 2, F)


	total_error = 0
	for i in range(len(undistorted_l)):
		total_error += abs(undistorted_l[i][0][0] *
						   lines_l[i][0][0] +
						   undistorted_l[i][0][1] *
						   lines_r[i][0][1] +
						   lines_r[i][0][2])
						   
	total_points = idx * len(objpoints)
	#print(total_error/total_points)

	total_error = 0
	for i in range(len(undistorted_r)):
		total_error += abs(undistorted_r[i][0][0] *
						   lines_r[i][0][0] +
						   undistorted_r[i][0][1] *
						   lines_l[i][0][1] +
						   lines_l[i][0][2])
						   
	total_points = idx * len(objpoints)
	#print(total_error/total_points)

	#img1, img2 = drawlines(imgL,imgR,lines,pts1,pts2)

	for line in range(0, int(imgR.shape[0] / 20)):
			# img1[line * 20, :] = ((20-i)*5, i*2, i*10)
			# img2[line * 20, :] = ((20-i)*5, i*2, i*10)
			dst_l[line * 20, :] = ((20-i)*5, i*5, i*10)
			dst_r[line * 20, :] = ((20-i)*5, i*5, i*10)
			i = i + 1
			
	cv.imshow('image',np.hstack((imgR,imgL)))
	cv.imshow('image-after',np.hstack((dst_r,dst_l)))
	print(Q)
	np.savez("full_stereo_calib.npz", K1= mtx_l, K2=cameraMatrix2, D1=dist_l, D2=dist_r ,R1=rect_l,
				R2=rect_r, P1=proj_l, P2=proj_r, Q=Q)
	cv.waitKey()