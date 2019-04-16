# organize imports
import cv2
import numpy as np
import os

IMG_SIZE=96

# region of interest (ROI) coordinates
top, right, bottom, left = 100, 150, 400, 450

exit_con='**'

a=''

dir0=input('enter the directory name : ')

try:
    os.mkdir(dir0)
except:
    print('contain folder in same name')

# get the reference to the webcam
camera = cv2.VideoCapture(0)

while(True):

    a=input('exit: ** or enter the label name : ')

    if a==exit_con:
        break

    dir1=str(dir0)+'/'+str(a)
    print(dir1)

    try:
        os.mkdir(dir1)
    except:
        print('contain folder')

    i=0

    while(True):
        (t, frame) = camera.read()

        

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        #resize img
        gray = cv2.resize(gray, (IMG_SIZE,IMG_SIZE))

        #write img file to directory
        cv2.imwrite("%s/%s/%d.jpg"%(dir0,a,i),gray)
        i+=1
        print(i)
        if i>500:
            break

        # draw the segmented hand
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        cv2.imshow("Video Feed 1", gray)

        cv2.imshow("Video Feed", frame)
        # observe the keypress by the user
        keypress = cv2.waitKey(1)

        # if the user pressed "Esc", then stop looping
        if keypress == 27:
            break

# free up memory
camera.release()
cv2.destroyAllWindows()

    


