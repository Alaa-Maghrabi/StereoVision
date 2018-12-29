## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2 as cv

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Intrinsics & Extrinsics
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

        hsv = cv.cvtColor(color_image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, (0, 100, 20), (20, 255, 255))
        mask = cv.erode(mask, None, iterations=1)
        mask = cv.dilate(mask, None, iterations=3)
        cv.imshow('mask', mask)

        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        x, xc = -999, -999
        y, yc = -999, -999
        radius = -999

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(c)
            # computing the centroid of the ball
            M = cv.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv.circle(color_image, (int(x), int(y)), int(radius), (0, 255, 255), 1)
                cv.circle(color_image, center, 2, (0, 255, 255), -1)

                xc = int(x)
                yc = int(y)
                depth = depth_frame.get_distance(xc, yc)
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [xc, yc], depth)
        # print(depth_point)
        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
        cv.imshow(' ', images)
        cv.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()