
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import cnn_sgn

IMG_SIZE = 96
LR = 1e-3

nb_classes=28

MODEL_NAME = 'handsign.model'



model=cnn_sgn.cnn_model()

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')




# organize imports
import cv2
import imutils
import numpy as np

from collections import Counter

import time



            #0    1    2     3    4      5    6    7        8     9    10   11   12   13   14    15    16   17   18   19   20   21   22   23   24    25    26   27  28
out_label=['U', 'T', 'E', 'R', 'BKSP', 'Q', 'D', 'BLNK1', 'G', 'SPC', 'I', 'F', 'O', 'C', 'W', 'Y', 'BLNK', 'V', 'H', 'H', 'P', 'A', 'S', 'L', 'K', 'X', 'N', 'B', ]


pre=[]

s=''
cchar=[0,0]
c1=''

# initialize weight for running average
aWeight = 0.5

# get the reference to the webcam
camera = cv2.VideoCapture(0)

# region of interest (ROI) coordinates
top, right, bottom, left = 170, 150, 425, 450

# initialize num of frames
num_frames = 0

flag=0
flag1=0

# keep looping, until interrupted
while(True):
    # get the current frame
    (grabbed, frame) = camera.read()

    # resize the frame
    frame = imutils.resize(frame, width=700)

    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)

    # clone the frame
    clone = frame.copy()

    # get the height and width of the frame
    (height, width) = frame.shape[:2]

    # get the ROI
    roi = frame[top:bottom, right:left]

    # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    
    # cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
    
    img=gray
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img=cv2.imread("240fn.jpg",cv2.IMREAD_GRAYSCALE)
    # img=cv2.cvtColor(bw_image,cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    test_data =img

    orig = img
    data = img.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    # print(model_out)
    model_out = model.predict([data])[0]
        # print(model_out)
    pnb=np.argmax(model_out)
    print(str(np.argmax(model_out))+" "+str(out_label[pnb]))

    pre.append(out_label[pnb]) 


    cv2.putText(clone,
           '%s ' % (str(out_label[pnb])),
           (450, 150), cv2.FONT_HERSHEY_PLAIN,5,(0, 255, 0))

            

            
        


    # draw the segmented hand
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

    cv2.putText(clone,
                   '%s ' % (str(s)),
                   (10, 60), cv2.FONT_HERSHEY_PLAIN,3,(0, 0, 0))

    # increment the number of frames
    num_frames += 1
    # time.sleep(.3)
    # display the frame with segmented hand
    cv2.imshow("Video Feed", clone)

    # observe the keypress by the user
    keypress = cv2.waitKey(1) & 0xFF

    # if the user pressed "q", then stop looping
    if keypress == ord("q"):
        break

    elif keypress == 27:
        break

# free up memory
camera.release()
cv2.destroyAllWindows()
