import cv2

cam = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cam.set(3, 640)
cam.set(4, 480)
cap1.set(3, 640)
cap1.set(4, 480)
cv2.waitKey(1000)

img_counter = 0

while True:

	ret, frame = cam.read()
	ret1, frame1 = cap1.read()
	
	cv2.imshow("left", frame)
	cv2.imshow("right", frame1)
	
	if not ret:
		break
	k = cv2.waitKey(1)

	if k%256 == 27:
		# ESC pressed
		print("Escape hit, closing...")
		break
	elif k%256 == 32:
        # SPACE pressed
		img_name = "left_{}.png".format(img_counter)
		cv2.imwrite(img_name, frame)
		img_name1 = "right_{}.png".format(img_counter)
		cv2.imwrite(img_name1, frame1)
		print("{} written!".format(img_name))
		img_counter += 1

cam.release()
cap1.release()

cv2.destroyAllWindows()