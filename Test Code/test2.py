import cv2 as cv
import numpy as np

cap = cv.VideoCapture('../Project_1_AR_tag_detection/Tag0.mp4')

while(cap.isOpened()):
	ret, frame = cap.read()

	#img = cv.imread(filename)
	gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

	#gray = np.float32(gray)
	# dst = cv.cornerHarris(gray,4,15,0.04)
	# # cornerHarris(gray, blockSize, ksize, k)
	# # Reducing k increases the number of corners detected
	# # Reducing blockSize decreases the number of corners detected
	# # 

	# #result is dilated for marking the corners, not important
	# dst = cv.dilate(dst,None)

	# # Threshold for an optimal value, it may vary depending on the image.
	# frame[dst>0.2*dst.max()]=[0,0,255]
	# # Reducing threshold increases number of corners detected
	ret, thresh = cv.threshold(gray, 200, 255, 0)
	contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	cv.drawContours(frame, contours[1], -1, (0,255,0), 3)
	#cv.imshow('', thresh)
	cv.imshow('dst',frame)
	#print(contours)
	if cv.waitKey(): #& 0xff == 27:
	 	break
	 	cv.destroyAllWindows()