import cv2 as cv
import numpy as np


def getTagInfoMat(image):
	(height, width) = image.shape
	cellHeight = int(np.floor(height/8))
	cellWidth = int(np.floor(width/8))
	cellTotal = cellHeight*cellWidth
	infoMat = np.zeros((8,8),np.float32)
	for i in range(8):
		for j in range(8):
			cellSum = 0
			for h_iter in range(i*cellHeight, ((i+1)*cellHeight)-1):
				for w_iter in range(j*cellWidth, ((j+1)*cellWidth)-1):
					cellSum += image[h_iter][w_iter]
			if (cellSum/cellTotal > 150):
				infoMat[i][j] = 1

	return infoMat


def getTagOrientation(infoMat):
	if infoMat[2][2] == 1:
		orientation = 'down'
	elif infoMat[5][2] == 1:
		orientation = 'right'
	elif infoMat[5][5] == 1:
		orientation = 'up'
	else:
		orientation = 'left'

	return orientation



def getTagID(infoMat):
	if getTagOrientation(infoMat) == 'down':
		ID = (2**0)*infoMat[4][4] + (2**1)*infoMat[4][3] + (2**2)*infoMat[3][3] + (2**3)*infoMat[3][4]
	elif getTagOrientation(infoMat) == 'right':
		ID = (2**0)*infoMat[3][4] + (2**1)*infoMat[4][4] + (2**2)*infoMat[4][3] + (2**3)*infoMat[3][3]
	elif getTagOrientation(infoMat) == 'up':
		ID = (2**0)*infoMat[3][3] + (2**1)*infoMat[3][4] + (2**2)*infoMat[4][4] + (2**3)*infoMat[4][3]
	else:
		ID = (2**0)*infoMat[4][3] + (2**1)*infoMat[3][3] + (2**2)*infoMat[3][4] + (2**3)*infoMat[4][4]

	return ID



# This function reads the b/w image and returns the indices of those contours that have
# at least one child and exactly one parent. This condition translates to the tag outline
# in the videos.
def getTagBoundaryContour(hierarchy):
	indList = []
	for index, element in enumerate(hierarchy[0]):
		if (element[2]!=-1) and (element[3]!=-1) and (hierarchy[0][element[3]][3]==-1):
			indList.append(index)

	return indList


def getCorners(contour):
	rect = [[], [], [], []]
	maxSum = [None, 0]
	minSum = [None, np.inf]
	minDIff = [None, np.inf]
	maxDiff = [None, 0]
	minX = [None, np.inf]
	maxX = [None, 0]
	minY = [None, np.inf]
	maxY = [None, 0]
	
	for index in range(len(contour)):
		currSum = contour[index][0][0]+contour[index][0][1]
		currDiff = contour[index][0][0]-contour[index][0][1]
		if currSum > maxSum[1]:
			maxSum = [index, currSum]
		if currSum < minSum[1]:
			minSum = [index, currSum]
		if currDiff > maxDiff[1]:
			maxDiff = [index, currDiff]
		if currDiff < minDIff[1]:
			minDIff = [index, currDiff]
		if contour[index][0][0] > maxX[1]:
			maxX = [index, contour[index][0][0]]
		if contour[index][0][1] > maxY[1]:
			maxY = [index, contour[index][0][1]]
		if contour[index][0][0] < minX[1]:
			minX = [index, contour[index][0][0]]
		if contour[index][0][1] < maxY[1]:
			minY = [index, contour[index][0][1]]

	rect[0] = contour[minSum[0]][0]
	rect[1] = contour[maxDiff[0]][0]
	rect[2] = contour[maxSum[0]][0]
	rect[3] = contour[minDIff[0]][0]

	# Guard condition for when the above method fails. Typically happens when the inclination
	# of any of the sides is close to +/-45 degrees.
	if (abs((rect[0][1] - rect[2][1])) < 50):
		rect[0] = contour[minX[0]][0]
		rect[1] = contour[minY[0]][0]
		rect[2] = contour[maxX[0]][0]
		rect[3] = contour[maxY[0]][0]

	return np.array(rect,np.float32)



cap = cv.VideoCapture('../Project_1_AR_tag_detection/Video_dataset/Tag2.mp4')

while(cap.isOpened()):
	ret, frame = cap.read()

	gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

	# Thresholding to improve contour detection
	ret, thresh = cv.threshold(gray, 200, 255, 0)

	contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	# Draw each contour
	for index in getTagBoundaryContour(hierarchy):
		# cv.drawContours(frame, contours[index], -1, (0,0,255), 2)
		corners = getCorners(contours[index])
		# for point in corners:
		# 	cv.circle(frame,(point[0],point[1]),10,(0,0,255))
		width = max(np.linalg.norm(corners[0]-corners[1]), np.linalg.norm(corners[2]-corners[3]))
		height = max(np.linalg.norm(corners[2]-corners[1]), np.linalg.norm(corners[0]-corners[3]))
		final = np.array([[0,0], [width-1,0],[width-1,height-1],[0,height-1]],np.float32)
		M = cv.getPerspectiveTransform(corners, final)
		warped = cv.warpPerspective(frame, M, (width, height))
		cv.imshow('tag',warped)
		infoMat = getTagInfoMat(cv.cvtColor(warped,cv.COLOR_BGR2GRAY))
		print(getTagID(infoMat))

	#cv.imshow('dst',frame)

	# Runs till the end of the video
	if cv.waitKey():# & 0xff == 27:
	 	break
	 	cv.destroyAllWindows()