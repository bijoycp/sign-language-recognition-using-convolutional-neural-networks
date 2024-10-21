# sign-language-recognition-using-convolutional-neural-networks
## sign language recognition using convolutional neural networks tensorflow tflean opencv and python

### Software Specification<br />
tensorflow version : 1.4.0<br />
opencv : 3.4.0<br />
numpy : 1.15.4<br />

## install packages
sudo apt-get update  
sudo apt-get upgrade  
sudo apt install htop  
sudo apt-get install python-pip  
sudo apt install python3-pip --reinstall  
pip3 install opencv-contrib-python  
pip3 install tensorflow==1.4.0  
pip3 install tflearn  
pip3 install talkey  
sudo apt-get install espeak  
pip3 install imutils  
sudo pip3 install  python-dateutil==2.5.0  

 ## Run the Code  
 ### Capture the Image 
 ```
 python3 1_img_cap.py  
 ```
 ### Create Dataset 
 ```
 python3 2_create_dataset.py  
 ```
 ### Train the Model 
 ```
 python3 3_cnn_train.py  
 ```
 ### Display Output 
 ```
 python3 4_cam_cnn_out.py  
 python3 5_cam_cnn_out_string.py  
 ```
 
 ## Output of the Project
### Video
### Ubuntu
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/3TOiZiPHpTU/0.jpg)](https://www.youtube.com/watch?v=3TOiZiPHpTU)
### Raspberry Pi
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/0Cv16PBM4Ro/0.jpg)](https://www.youtube.com/watch?v=0Cv16PBM4Ro)
 ## Final output 
image filters like threshold and other filters are not used for creating the dataset  
the model was trained using 96x96 grayscale images

![output](https://user-images.githubusercontent.com/18006433/57105815-708f7200-6d49-11e9-8a1b-b8aa525dc541.png)
## TensorBoard: Graph Visualization
in final program 5 hidden layers were used for increasing the accuracy of the model  
![graph](https://user-images.githubusercontent.com/18006433/57106555-56569380-6d4b-11e9-842c-b00f1558dcb0.png)
## Image capturing for Creating Dataset

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/mpdPXWcXp3I/0.jpg)](https://www.youtube.com/watch?v=mpdPXWcXp3I)

## Create Dataset
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/6H-YQlrgn6U/0.jpg)](https://www.youtube.com/watch?v=6H-YQlrgn6U)
