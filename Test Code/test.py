import numpy as np
import cv2
from matplotlib import pyplot as plt


cap = cv2.VideoCapture('Tag0.mp4')

while(cap.isOpened()):
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	corners = cv2.goodFeaturesToTrack(gray,4,0.01,48)
	corners = np.int0(corners)

	for i in corners:
		x,y = i.ravel()
		cv2.circle(gray,(x,y),3,(0, 0, 255),-1)

	plt.imshow(gray),plt.show()
	#cv2.imshow('frame',gray)
	if cv2.waitKey(): #& 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()