import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import warnings


def getHomographyCorners(contours):
	corners = []
	for c in contours:
		if (cv2.arcLength(c, True) > 268):
			if cv2.arcLength(c, True) < 500:
				maxX = 0
				minX = np.inf
				for point in c:
					if point[0][0] < minX:
						bottom = point[0]
						minX = point[0][0]
					elif point[0][0] > maxX:
						top = point[0]
						maxX = point[0][0]
			else:
				rightLane = c

	for point in rightLane:
		if point[0][1] == top[1]:
			rightTop = point[0]
		elif point[0][1] == bottom[1]:
			rightBottom = point[0]

	corners = [top, rightTop, rightBottom, bottom]
	return np.array(corners, np.float32)


def drawHistogram(thresh, nbins):
	count = []
	for x in range(thresh.shape[1]):
		for y in range(thresh.shape[0]):
			if thresh[y][x] == 255:
				count.append(x)

	#plt.plot(range(len(count)), count)
	ret = plt.hist(count, bins=nbins)
	#plt.show()

	return ret	


def fitPolynomial(thresh, lower, upper):
	dataset = []
	for x in range(int(lower), int(upper)):
		for y in range(thresh.shape[0]):
			if thresh[y][x] == 255:
				dataset.append([x,y])

	# print(dataset)
	matrix_y = []
	matrix_x = []
	for point in dataset:
	 	matrix_y.append(point[1])
	 	#matrix_x.append((point[0]**2, point[0], 1))
	 	matrix_x.append(point[0])
	y = np.array(matrix_y)
	x = np.array(matrix_x)

	with warnings.catch_warnings():
		warnings.filterwarnings('ignore')
		try:
			coeffs = np.polyfit(matrix_x, matrix_y, 2)
		except:
			coeffs = np.polyfit(matrix_x, matrix_y, 1)
	# xt = np.transpose(x)
	# B = np.matmul(np.linalg.inv(np.matmul(xt, x)), np.matmul(xt, y))

	points = []
	# for xi in x:
	# 	points.append((int(xi[1]), int(xi[0]*B[0] + xi[1]*B[1] + xi[2]*B[2])))

	#print(points)

	if len(coeffs) == 3:
		for xi in x:
			points.append((int(xi), int(coeffs[0]*(xi**2) + coeffs[1]*xi + coeffs[2])))
	else:
		for xi in x:
			points.append((int(xi), int(coeffs[0]*xi + coeffs[1])))

	return points


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------MAIN BODY------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
firstFrame = True
K_matrix = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02], [0.000000e+00, 9.019653e+02, 2.242509e+02], [0.000000e+00, 0.000000e+00, 1.000000e+00]], np.float32)
distCoeffs = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02], np.float32)


for file in sorted(glob.glob("../../../data_1/data/*.png")):
	#frame = cv2.imread('../../../data_1/data/0000000200.png')
	frame = cv2.imread(file)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	
	if firstFrame:
	# 	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# 	ret, thresh = cv2.threshold(gray, 254, 255, 0) # Thresholding to improve contour detection
	# 	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	# 	corners = getHomographyCorners(contours)
	# 	width = corners[1][0] - corners[0][0]
	# 	height = np.sqrt((corners[3][1]-corners[0][1])**2 + (corners[3][0]-corners[0][0])**2)
	# 	destCorners = np.array([corners[0], [corners[0][0]+width,corners[0][1]], [corners[0][0]+width,corners[0][1]+height], [corners[0][0],corners[0][1]+height]], np.float32)
	# 	H = cv2.getPerspectiveTransform(corners, destCorners)
	# 	firstFrame = False
	# 	offsetSize = 700
	# 	warped = np.zeros((int(height+offsetSize), int(width+offsetSize)), dtype=np.uint8)


		frame = cv2.undistort(frame, K_matrix, distCoeffs)
		corners = np.array([[570,260],[699,260],[950,420],[5,420]],np.float32)
		width = corners[1][0] - corners[0][0]
		height = np.sqrt((corners[3][1]-corners[0][1])**2 + (corners[3][0]-corners[0][0])**2)
		destCorners = np.array([[0,0],[width,0],[width,height],[0,height]],np.float32) #np.array([corners[0], [corners[0][0]+width,corners[0][1]], [corners[0][0]+width,corners[0][1]+height], [corners[0][0],corners[0][1]+height]], np.float32)
		H = cv2.getPerspectiveTransform(corners, destCorners)
		Hinv = cv2.getPerspectiveTransform(destCorners, corners)
		warped = np.zeros((int(height), int(width)), dtype=np.uint8)
		firstFrame = False

	#filtered = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21) 
	#filtered = cv2.GaussianBlur(frame,(7,7),0)
	#filtered = cv2.bilateralFilter(frame,13,75,75)
	warped = cv2.warpPerspective(frame, H, (warped.shape[1], warped.shape[0]))
	ret, thresh = cv2.threshold(cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY), 240, 255, 0)
	hist = drawHistogram(thresh, 20)
	#print(hist[1])

	for bar in range(len(hist[0])):
		if hist[0][bar] > 70:
			lowerBound = np.floor(hist[1][bar])
			upperBound = min(np.ceil(hist[1][bar+1]), thresh.shape[1])
			#print(lowerBound)
			points = fitPolynomial(thresh, lowerBound, upperBound)
			#newPoints = np.zeros((len(points), len(points[0])), np.float32)

			#cv2.perspectiveTransform(np.array(points,np.float32), newPoints, Hinv)
			for index in range(len(points)-1):
				#frame = cv2.line(frame, tuple(newPoints[index]), tuple(newPoints[index+1]), (0,0,255), thickness=3)
				warped = cv2.line(warped, points[index], points[index+1], (0,0,255), thickness=3)
				#warped = cv2.circle(warped, point, 1, (0,0,255), 2)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	cv2.imshow('warped', warped)
	#cv2.imshow('original', frame)
	if cv2.waitKey(10) & 0xff == 27:	
	 	break
	 	cv2.destroyAllWindows()