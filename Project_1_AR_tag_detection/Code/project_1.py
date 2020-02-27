import cv2
import numpy as np
import sys


def customWarp(inp, output, H, lims):
	H_inv = np.linalg.inv(H)
	H_inv = H_inv/H_inv[2][2]

	for i in range(lims[0][0], lims[0][1]):
		for j in range(lims[1][0], lims[1][1]):
			temp = np.array([i, j, 1]).T
			warp = np.matmul(H_inv, temp)
			warp = (warp/warp[2]).astype(int)

			if (warp[0] < inp.shape[0]) and (warp[1] < inp.shape[1]) and (warp[0] > -1) and (warp[1] > -1):
				output[j][i] = inp[warp[1]][warp[0]]	

	return output



def findHMatrix(pts1,pts2):
        A = np.zeros((8,9))

        for i in range(4):
        	A[2*i][0] = pts2[i-1][0]
        	A[2*i][1] = pts2[i-1][1]
        	A[2*i][2] = 1
        	A[2*i][6] = -pts1[i-1][0]*pts2[i-1][0]
        	A[2*i][7] = -pts1[i-1][0]*pts2[i-1][1]
        	A[2*i][8] = -pts1[i-1][0]
        	A[2*i+1][3] = pts2[i-1][0]
        	A[2*i+1][4] = pts2[i-1][1]
        	A[2*i+1][5] = 1
        	A[2*i+1][6] = -pts1[i-1][1]*pts2[i-1][0]
        	A[2*i+1][7] = -pts1[i-1][1]*pts2[i-1][1]
        	A[2*i+1][8] = -pts1[i-1][1]

        u, s, V = np.linalg.svd(A, full_matrices = True)

        a = []
        if V[8][8] == 1:
            for i in range(0,9):
                a.append(V[8][i])
        else:
            for i in range(0,9):
                a.append(V[8][i]/V[8][8])

        # H matrix in 3X3 shape
        b = np.reshape(a, (3, 3))
        return b 


# Returns projection matrix using intrinsic parameters and homography matrix
def getProjectionMatrix(Hmatrix, Kmatrix):
	K_inv = np.linalg.inv(Kmatrix)
	lam = 2/(np.linalg.norm(np.matmul(K_inv, Hmatrix[:,0])) + np.linalg.norm(np.matmul(K_inv, Hmatrix[:,1])))
	B_tilda = np.matmul(K_inv, Hmatrix)
	if (np.linalg.det(B_tilda) < 0):
		B = -lam*B_tilda
	else:
		B = lam*B_tilda

	r1 = lam*B[:,0]
	r2 = lam*B[:,1]
	r3 = np.cross(r1, r2)/lam
	t = np.array([lam*B[:,2]]).T
	R = np.array([r1,r2,r3]).T
	R = np.hstack([R, t])
	projMat = np.matmul(Kmatrix, np.matrix(R))
	projMat = projMat/projMat[2,3]

	return projMat


# Draws a cube of given height on top of the tag
def drawCube(projMat, height, corners_world, corners_image, image):

	# Upper corners of the cube in world coordinates
	p0 = np.array([corners_world[0][0], corners_world[0][1], -height, 1])
	p1 = np.array([corners_world[1][0], corners_world[1][1], -height, 1])
	p2 = np.array([corners_world[2][0], corners_world[2][1], -height, 1])
	p3 = np.array([corners_world[3][0], corners_world[3][1], -height, 1])

	# Upper corners of the cube in image coordinates
	p0_proj = np.matmul(projMat, p0.T)
	p1_proj = np.matmul(projMat, p1.T)
	p2_proj = np.matmul(projMat, p2.T)
	p3_proj = np.matmul(projMat, p3.T)

	edgeColor = (0, 0, 200)
	thickness = int(2)

	# Join all vertices of the cube
	cv2.line(image, tuple(corners_image[0]), tuple(corners_image[1]), edgeColor, thickness)
	cv2.line(image, tuple(corners_image[1]), tuple(corners_image[2]), edgeColor, thickness)
	cv2.line(image, tuple(corners_image[2]), tuple(corners_image[3]), edgeColor, thickness)
	cv2.line(image, tuple(corners_image[3]), tuple(corners_image[0]), edgeColor, thickness)

	cv2.line(image, tuple(corners_image[0]), (int(p0_proj[0,0]/p0_proj[0,2]), int(p0_proj[0,1]/p0_proj[0,2])), edgeColor, thickness)
	cv2.line(image, tuple(corners_image[1]), (int(p1_proj[0,0]/p1_proj[0,2]), int(p1_proj[0,1]/p1_proj[0,2])), edgeColor, thickness)
	cv2.line(image, tuple(corners_image[2]), (int(p2_proj[0,0]/p2_proj[0,2]), int(p2_proj[0,1]/p2_proj[0,2])), edgeColor, thickness)
	cv2.line(image, tuple(corners_image[3]), (int(p3_proj[0,0]/p3_proj[0,2]), int(p3_proj[0,1]/p3_proj[0,2])), edgeColor, thickness)

	cv2.line(image, (int(p0_proj[0,0]/p0_proj[0,2]), int(p0_proj[0,1]/p0_proj[0,2])), (int(p1_proj[0,0]/p1_proj[0,2]), int(p1_proj[0,1]/p1_proj[0,2])), edgeColor, thickness)
	cv2.line(image, (int(p1_proj[0,0]/p1_proj[0,2]), int(p1_proj[0,1]/p1_proj[0,2])), (int(p2_proj[0,0]/p2_proj[0,2]), int(p2_proj[0,1]/p2_proj[0,2])), edgeColor, thickness)
	cv2.line(image, (int(p2_proj[0,0]/p2_proj[0,2]), int(p2_proj[0,1]/p2_proj[0,2])), (int(p3_proj[0,0]/p3_proj[0,2]), int(p3_proj[0,1]/p3_proj[0,2])), edgeColor, thickness)
	cv2.line(image, (int(p3_proj[0,0]/p3_proj[0,2]), int(p3_proj[0,1]/p3_proj[0,2])), (int(p0_proj[0,0]/p0_proj[0,2]), int(p0_proj[0,1]/p0_proj[0,2])), edgeColor, thickness)

	return image


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
	# if (abs((rect[0][1] - rect[0][1])) < 50):
	# 	rect[0] = contour[minX[0]][0]
	# 	rect[1] = contour[minY[0]][0]
	# 	rect[2] = contour[maxX[0]][0]
	# 	rect[3] = contour[maxY[0]][0]

	try:
		return np.array(rect,np.float32)
	except:
		return [[None]]		# Takes care of the boundary condition when the tag moves outside the frame

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------MAIN BODY------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

cube = picture = False

if len(sys.argv) == 1:
	cube = True
	videoName = 'Tag0'

else:
	mode = sys.argv[1]

	if (mode == 'cube'):
		cube = True
		videoName = sys.argv[2]
	elif (mode == 'lena'):
		picture = True
		videoName = sys.argv[2]
	else:
		print("Expected two arguments:\n(1) Mode of operation (lena or cube)\n(2) Video Name (e.g. Tag0, or multipleTags)")
		exit(0)

if cube:
	Kmatrix = np.array([[1406.08415449821,0,0],[2.20679787308599, 1417.99930662800,0],[1014.13643417416, 566.347754321696,1]]).T

cap = cv2.VideoCapture('../Video_dataset/' + videoName + '.mp4')

lena = cv2.imread('../reference_images/images/Lena.png')
(lenaH, lenaW, c) = lena.shape
lenaCorners = np.array([[0,0], [0,lenaW], [lenaH,lenaW], [lenaH,0]],np.float32)

while(cap.isOpened()):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 240, 255, 0) # Thresholding to improve contour detection
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	masked = cv2.bitwise_and(frame, frame, mask = cv2.inRange(frame, 180, 255))

	# For each tag boundary contour detected
	for index in getTagBoundaryContour(hierarchy):
		corners = getCorners(contours[index])	# Tag corners
		if corners[0][0] == None: continue
		width = 100
		height = 100
		final = np.array([[0,0], [width-1,0],[width-1,height-1],[0,height-1]],np.float32) # the corners for homographic transformation
		Homography = findHMatrix(corners, final)
		warped = customWarp(masked, np.zeros((width, height)), Homography, [[0, width], [0, height]])
		warped = warped[0:width][0:height]

		infoMat = getTagInfoMat(warped)		# Get encoded tag data
		newCorners = reorientCorners(corners, getTagOrientation(infoMat))	# Re-orient corners based on detected tag orientation
		cv2.putText(frame, 'Tag' + str(getTagID(infoMat)), (int(corners[0][0]-10), int(corners[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,200), 2, cv2.LINE_AA)

		if picture:
			lims = np.array([[min(corners[:,0]), max(corners[:,0])], [min(corners[:,1]), max(corners[:,1])]]).astype(int)
			persp = findHMatrix(newCorners, lenaCorners)
			frame = customWarp(lena, frame, persp, lims)
			
		
		if cube:
			persp = findHMatrix(newCorners, lenaCorners)
			projMat = getProjectionMatrix(persp, Kmatrix)
			frame = drawCube(projMat, 500, lenaCorners, newCorners, frame)

	cv2.imshow('dst',frame)

	# Runs till the end of the video
	if cv2.waitKey(30) & 0xff == 27:	
	 	break
	 	cv2.destroyAllWindows()