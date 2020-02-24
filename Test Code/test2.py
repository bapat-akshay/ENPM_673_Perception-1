import cv2 as cv
import numpy as np

# Generate a new order of corners according to the detected orientation of the tag
def reorientCorners(corners, orientation):
	newCorners = np.array([[0,0], [0,0], [0,0], [0,0]],np.float32)
	if (orientation == 'down'):
		newCorners[0] = [corners[3][0], corners[3][1]]
		newCorners[1] = [corners[0][0], corners[0][1]]
		newCorners[2] = [corners[1][0], corners[1][1]]
		newCorners[3] = [corners[2][0], corners[2][1]]
	elif (orientation == 'right'):
		newCorners[0] = [corners[2][0], corners[2][1]]
		newCorners[1] = [corners[3][0], corners[3][1]]
		newCorners[2] = [corners[0][0], corners[0][1]]
		newCorners[3] = [corners[1][0], corners[1][1]]
	elif (orientation == 'up'): 
		newCorners[0] = [corners[1][0], corners[1][1]]
		newCorners[1] = [corners[2][0], corners[2][1]]
		newCorners[2] = [corners[3][0], corners[3][1]]
		newCorners[3] = [corners[0][0], corners[0][1]]
	else:
		newCorners[0] = [corners[0][0], corners[0][1]]
		newCorners[1] = [corners[1][0], corners[1][1]]
		newCorners[2] = [corners[2][0], corners[2][1]]
		newCorners[3] = [corners[3][0], corners[3][1]]

	return newCorners


# This function returns the 8x8 matrix that encodes the tag data
def getTagInfoMat(image):
	(height, width) = image.shape
	cellHeight = int(np.floor(height/8))
	cellWidth = int(np.floor(width/8))
	offset = int(cellHeight*0.1)
	cellTotal = (cellHeight-offset)*(cellWidth-offset)
	if not cellTotal:
		cellTotal = 1
	infoMat = np.zeros((8,8),np.float32)
	maximum = [0, None, None]
	for i in range(8):
		for j in range(8):
			cellSum = 0
			for h_iter in range(i*cellHeight + offset, ((i+1)*cellHeight)-1-offset):
				for w_iter in range(j*cellWidth+offset, ((j+1)*cellWidth)-1-offset):
					cellSum += image[h_iter][w_iter]
			if ((i == 2) and (j == 2)) or ((i == 2) and (j == 5)) or ((i == 5) and j == 5) or ((i == 5) and (j == 2)):
				if (cellSum/cellTotal > maximum[0]):
					maximum = [cellSum/cellTotal, i, j]
			elif (cellSum/cellTotal > 200):
				infoMat[i][j] = 1

	infoMat[maximum[1]][maximum[2]] = 1

	return infoMat


# Self-explanatory
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



# Get tag ID from decoded tag data
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


# Detect 4 corners of the contour that represents the tag outline
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
	if (abs((rect[0][0] - rect[2][0])) < 50):
		rect[0] = contour[minX[0]][0]
		rect[1] = contour[minY[0]][0]
		rect[2] = contour[maxX[0]][0]
		rect[3] = contour[maxY[0]][0]

	return np.array(rect,np.float32)



cap = cv.VideoCapture('../Project_1_AR_tag_detection/Video_dataset/multipleTags.mp4')

lena = cv.imread('../Project_1_AR_tag_detection/reference_images/images/Lena.png')
(lenaH, lenaW, c) = lena.shape
lenaCorners = np.array([[0,0], [0,lenaW], [lenaH,lenaW], [lenaH,0]],np.float32)

while(cap.isOpened()):
	ret, frame = cap.read()
	gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
	ret, thresh = cv.threshold(gray, 240, 255, 0) # Thresholding to improve contour detection
	contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	# Draw each contour
	for index in getTagBoundaryContour(hierarchy):
		corners = getCorners(contours[index])
		width = 100
		height = 100
		final = np.array([[0,0], [width-1,0],[width-1,height-1],[0,height-1]],np.float32) # the corners for homographic transformation
		M = cv.getPerspectiveTransform(corners, final)
		warped = cv.warpPerspective(frame, M, (width, height))
		#cv.imshow('tag',warped)
		infoMat = getTagInfoMat(cv.cvtColor(warped,cv.COLOR_BGR2GRAY))
		ID = getTagID(infoMat)
		#x = corners[0][0]
		#y = corners[0][1]
		#cv.putText(frame, 'Tag #' + str(ID) + ' orientation: ' + getTagOrientation(infoMat), (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
		#print(infoMat)
		newCorners = reorientCorners(corners, getTagOrientation(infoMat))
		persp = cv.getPerspectiveTransform(newCorners, lenaCorners)
		newWarped = cv.warpPerspective(lena, persp, (frame.shape[1], frame.shape[0]), dst=frame, flags=cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_TRANSPARENT)

	cv.imshow('dst',frame)

	# Runs till the end of the video
	if cv.waitKey(100) & 0xff == 27:
	 	break
	 	cv.destroyAllWindows()