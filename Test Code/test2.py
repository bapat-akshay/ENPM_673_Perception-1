import cv2
from cv2 import features2d
import numpy as np

cap = cv2.VideoCapture('Tag0.mp4')

while(cap.isOpened()):
	ret, frame = cap.read()

	#img = cv2.imread(filename)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	gray = np.float32(gray)
	# dst = cv2.cornerHarris(gray,4,15,0.04)
	# # cornerHarris(gray, blockSize, ksize, k)
	# # Reducing k increases the number of corners detected
	# # Reducing blockSize decreases the number of corners detected
	# # 

	# #result is dilated for marking the corners, not important
	# dst = cv2.dilate(dst,None)

	# # Threshold for an optimal value, it may vary depending on the image.
	# frame[dst>0.2*dst.max()]=[0,0,255]
	# # Reducing threshold increases number of corners detected

	sift = features2d.xfeatures2d.SIFT_create()
	kp = sift.detect(gray, None)
	frame = cv2.drawKeyPoints(gray, kp, frame)
	cv2.imwrite('sift.jpg', frame)

	# cv2.imshow('dst',frame)
	if cv2.waitKey(): #& 0xff == 27:
		break
		cv2.destroyAllWindows()