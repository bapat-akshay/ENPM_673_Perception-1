# Problem 1

import cv2

# Opens the Video file
cap = cv2.VideoCapture('/home/kulbir/Desktop/ENPM_673_Perception/Project_2_Lane_detection/Dataset_for_Problem_1/Night Drive - 2689.mp4')

#to save the video
#night_ride is the instance of VideoWriter
night_ride = cv2.VideoWriter('night_ride.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), 30, (1920, 1080))

#to check if cap has some value
# print(cap.isOpened())
imgArray = []
while(cap.isOpened()):
    ret, frame = cap.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if ret == True:
        # Uncomment To find the frame width and height:-
        # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # We use contrast limited adaptive histogram equalization , CLAHE
        # clip limit is the threshold for contrast limiting
        # tile grid size is the size of grid for histogram equalization
        Red, Green, Blue = cv2.split(frame)
        clahe = cv2.createCLAHE(clipLimit=11.0, tileGridSize=(3,3))

        clahe_output_red = clahe.apply(Red)
        clahe_output_green = clahe.apply(Green)
        clahe_output_blue = clahe.apply(Blue)

        frame_clahe = cv2.merge((clahe_output_red, clahe_output_green, clahe_output_blue))


        cv2.imshow('frame_new', frame_clahe)
        cv2.imshow('frame_old', frame)
        imgArray.append(frame_clahe)


        #to quit video
        if cv2.waitKey(30) & 0xff == ord('q'):
            break


    else:
        break
for i in range(len(imgArray)):
    night_ride.write(imgArray[i])

cap.release()
night_ride.release()
cv2.destroyAllWindows()
# Runs till the end of the video

