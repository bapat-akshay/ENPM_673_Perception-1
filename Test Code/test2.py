import cv2 as cv
import numpy as np


# This function reads the b/w image and returns the indices of those contours that have
# at least one child and exactly one parent. This condition translates to the tag outline
# in the videos.
def getTagBoundaryContour(hierarchy):
	indList = []
	for index, element in enumerate(hierarchy[0]):
		if (element[2]!=-1) and (element[3]!=-1) and (hierarchy[0][element[3]][3]==-1):
			indList.append(index)

	return indList


cap = cv.VideoCapture('../Project_1_AR_tag_detection/Video_dataset/multipleTags.mp4')

while(cap.isOpened()):
	ret, frame = cap.read()

	gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

	# Thresholding to improve contour detection
	ret, thresh = cv.threshold(gray, 200, 255, 0)
	
	contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

	# Draw each contour
	for index in getTagBoundaryContour(hierarchy):
		cv.drawContours(frame, contours[index], -1, (0,0,255), 2)

	cv.imshow('dst',frame)

	# Runs till the end of the video
	if cv.waitKey(1) & 0xff == 27:
	 	break
	 	cv.destroyAllWindows()