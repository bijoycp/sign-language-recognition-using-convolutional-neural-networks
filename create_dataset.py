import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.

path='train_data'

IMG_SIZE = 96

def create_train_data():
    training_data = []
    label=0
    for (dirpath,dirnames,filenames) in os.walk(path,topdown=True):
        print(dirpath)
        print(dirnames)
        # print(filenames)
        for dirname in dirnames:
            for(direcpath,direcnames,files) in os.walk(path+"/"+dirname,topdown=True):
                for file in files:
                    actual_path=path+"/"+dirname+"/"+file
                    # print(actual_path)
                    img=cv2.imread(actual_path,cv2.IMREAD_GRAYSCALE)                     
                    print(img)
                    

                    

create_train_data()