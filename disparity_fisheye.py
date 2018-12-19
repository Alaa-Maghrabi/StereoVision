import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
import orbslam2
import os.path

class Stereo:
	
	def __init__(self, path):
	# Define the paramateres for the disparity map, the calibration rectification
	# and other values
	
		self.window_size = 8
		self._minDisparity=0
		self.a=8            # max_disp 
		self._blockSize=4
		self._disp12MaxDiff=50
		self._uniquenessRatio=3
		self._speckleWindowSize=5
		self._speckleRange=2
		self._preFilterCap=55
		data = np.load(path)
		self.K_l = data['K1']
		self.K_r = data['K2']
		self.D_l = data['D1']
		self.D_r = data['D2']
		self.R_l = data['R1']
		self.R_r = data['R2']
		self.P_l = data['P1']
		self.P_r = data['P2']
		self.Q = data['Q']
		self.h, self.w = 480, 640
		self.map1l, self.map2l = [], []
		self.map1r, self.map2r = [], []
		self.left_matcher = []
		self.slam = []
		self.out = []
		
	def Initialize_mapping_calibration(self, disparity_bool = True, slam_bool=False, file_capture = False):
	# Initialize the mapping and the disparity matcher, to be called once and outside the loop
		
		#The mapping for the correction, careful always give the left image to the left parameters
		# and the right to the right parameters
		self.map1l, self.map2l = cv.fisheye.initUndistortRectifyMap(self.K_l, self.D_l, self.R_l, self.P_l, (self.w,self.h), cv.CV_32FC1)
		self.map1r, self.map2r = cv.fisheye.initUndistortRectifyMap(self.K_r, self.D_r, self.R_r, self.P_r, (self.w,self.h), cv.CV_32FC1)
		
		if disparity_bool:
			# The accuracy and the range of the disparity depends on these parameters
			self.left_matcher = cv.StereoSGBM_create(
				minDisparity=-self._minDisparity,
				numDisparities=16*self.a,             # max_disp has to be dividable by 16 f. E. HH 192, 256
				blockSize=self._blockSize,
				P1=8 * 3 * self.window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
				P2=32 * 3 * self.window_size ** 2,
				disp12MaxDiff=self._disp12MaxDiff,
				uniquenessRatio=self._uniquenessRatio,
				speckleWindowSize=self._speckleWindowSize,
				speckleRange=self._speckleRange,
				preFilterCap=self._preFilterCap,
				mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
				
			#Though this seems unecessary, it lowers the computation of the disparity map
			wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
		
		if slam_bool:
			vocab_path="Parameters/ORBvoc.txt"
			settings_path="EuRoC.yaml"
			self.slam = orbslam2.System(vocab_path, settings_path, orbslam2.Sensor.STEREO)
			self.slam.set_use_viewer(True)
			self.slam.initialize()
		
		if file_capture:
			f = open("Data.txt", "w+")
		
	def detect_ball(self, imageleft, show = False):
	# Part of the code to track the ball
	
		hsv = cv.cvtColor(imageleft, cv.COLOR_BGR2HSV)
   
		mask = cv.inRange(hsv, (0, 100, 20), (10, 255, 255)) 
		mask = cv.erode(mask,  None, iterations=1)
		mask = cv.dilate(mask, None, iterations=2)
		
		cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
								cv.CHAIN_APPROX_SIMPLE)[-2]
		center = None

		x, xc = 0, 0
		y, yc = 0, 0
		radius = 0

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
				if show:
					cv.circle(imageleft, (int(x), int(y)), int(radius), (0, 255, 255), 1)
					cv.circle(imageleft, center,           2, (0, 255, 255),  -1)

				xc = int(x)
				yc = int(y)
		
		if show:
			cv.imshow('Image',imageleft)
				
		return xc, yc, int(radius)
	
	def transform_disp_3d(self, xc, yc, disparity, start):
	# Here is the code that transforms the disparity and 2d coordinates to 3d coordinates
	# we can alwazs use cv.reprojectImageTo3D(displ, Q) but we are only interested in 
	# one point
		R = np.array([[xc],
						[yc],
						[disparity],
						[1]])
		point_3d = (self.Q).dot(R)
		image_3d = point_3d[0:3]/point_3d[3]
		sxyz = []
		
		# This line is only here to reduce the big outliers and can be omitted
		if(image_3d[2] < 2000 and image_3d[2] > 100):
			t = time.time()
			seconds = t - start
			sxyz = [seconds, image_3d[0,0],image_3d[1,0],image_3d[2,0]]
		
		return sxyz
	
	def collect_single_frame_data(self, left_frame, right_frame, start, show = False, file_capture = False):
	# This function gets the 3d coordinates of the basketball for one frame, to be called inside a loop
		
		# Start by rectifying the images
		imgL = cv.remap(left_frame, self.map1l, self.map2l, interpolation=cv.INTER_LINEAR)
		imgR = cv.remap(right_frame, self.map1r, self.map2r, interpolation=cv.INTER_LINEAR)
		imageleft=imgL.copy()
		imgL=cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
		imgR=cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
		
		# Calculate the disparity map
		displ = self.left_matcher.compute(imgL, imgR).astype(np.float32)/16
		displ = np.int16(displ)
		
		# Get the coordinates of the ball
		xc, yc, _ = self.detect_ball(imageleft, show)
		
		# Transform these 2d coordinates into 3d
		sxyz = self.transform_disp_3d( xc, yc, displ[yc, xc], start)
		
		# if we want to save the values in a file
		if file_capture:
			f.write("{}, {}, {}, {} \n".format(sxyz[0], sxyz[1], sxyz[2], sxyz[3]))
			
		#print(("{}, {}, {}, {} \n".format(sxyz[0], sxyz[1], sxyz[2], sxyz[3])))
		if sxyz:
			(self.out).append([sxyz[0], sxyz[1], sxyz[2], sxyz[3]])
		
	def plot_charts(self):
	# Funtion to plot the 3d coordinates
		self.out = np.array(self.out)
		plt.figure('time')
		plt.plot(self.out[:,0], label='3D')
		plt.legend(loc='upper left')
		plt.figure('x cm')
		plt.plot(self.out[:,1], label='3D')
		plt.legend(loc='upper left')
		plt.figure('y cm')
		plt.plot(self.out[:,2], label='3D')
		plt.legend(loc='upper left')
		plt.figure('z cm')
		plt.plot(self.out[:,3], label='3D')
		plt.legend(loc='upper left')
		plt.show()
	
	def destroy_feed(self, capture_left, capture_right, file_capture = False):
	# release the cameras and destroy the windows opened
		if file_capture:
			f.close() 
		
		capture_left.release()
		capture_right.release()
		cv.destroyAllWindows()
		
	def collect_frames_data(self, capture_left, capture_right, num_frames, show = False, file_capture = False):
	# This part of the code give you the matrix of 3d coordinates of the ball for X frames
	
		# Start by initializing the mapping and disparity
		self.Initialize_mapping_calibration(disparity_bool = True, slam_bool=False)
		
		# Start a counter to measure fps
		start = time.time()
		
		
		for frames in range(num_frames):
			ret, frame_left = capture_left.read()
			ret1, frame_right = capture_right.read()
			self.collect_single_frame_data(frame_left, frame_right, start, show, file_capture)
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
		
		return self.out
		
	def save_trajectory(self, filename):
		with open(filename, 'w') as traj_file:
			traj_file.writelines('{time} {r00} {r01} {r02} {t0} {r10} {r11} {r12} {t1} {r20} {r21} {r22} {t2}\n'.format(
				time=repr(t),
				r00=repr(r00),
				r01=repr(r01),
				r02=repr(r02),
				t0=repr(t0),
				r10=repr(r10),
				r11=repr(r11),
				r12=repr(r12),
				t1=repr(t1),
				r20=repr(r20),
				r21=repr(r21),
				r22=repr(r22),
				t2=repr(t2)
			) for t, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 in self.slam.get_trajectory_points())
	
	def SLAM_single_cycle(self, capture_left, capture_right, start):
		t = time.time()
		seconds = t - start
		tframe = seconds
		
		# Start by rectifying the images
		imgL = cv.remap(left_frame, self.map1l, self.map2l, interpolation=cv.INTER_LINEAR)
		imgR = cv.remap(right_frame, self.map1r, self.map2r, interpolation=cv.INTER_LINEAR)

		t1 = time.time()
		self.slam.process_image_stereo(imgL, imgR, tframe)
		t2 = time.time()

		ttrack = t2 - t1
		 return ttrack
		
	def SLAM(self, capture_left, capture_right, num_frames):

		# Start by initializing the mapping and disparity
		self.Initialize_mapping_calibration(disparity_bool = False, slam_bool=True)		
		
		# Start a counter to measure fps
		start = time.time()
		
		timestamps = []
		
		times_track = [0 for _ in range(num_frames)]
		print('-----')
		print('Start processing sequence ...')

		for idx in range(num_frames):
			ret, frame_left = capture_left.read()
			ret1, frame_right = capture_right.read()
			times_track[idx] = self.SLAM_single_cycle(capture_left, capture_right, start)
			
		self.save_trajectory('trajectory.txt')
		self.slam.shutdown()

		times_track = sorted(times_track)
		total_time = sum(times_track)
		print('-----')
		print('median tracking time: {0}'.format(times_track[num_frames // 2]))
		print('mean tracking time: {0}'.format(total_time / num_frames))

		return 0
	
if __name__ == '__main__':
# Here is an example of how to run the code to get the coordinates
	num_frames = 600
	capture_left = cv.VideoCapture(0)
	capture_right = cv.VideoCapture(1)

	disparity_map = Stereo('Parameters/fish_final_calib.npz')
	#out = disparity_map.collect_frames_data(capture_left, capture_right, num_frames, show= False)
	#disparity_map.plot_charts()
	disparity_map = self.SLAM(capture_left, capture_right, num_frames)
	disparity_map.destroy_feed(capture_left, capture_right)
	
	