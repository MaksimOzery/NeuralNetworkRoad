# -*- coding: cp1251 -*-
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D,Activation, Dropout
from keras.models import Model, Sequential
from keras.models import load_model,model_from_json
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import Adam
from skimage.io import imshow
import cv2,os
import numpy as np
import matplotlib.pyplot as plt
import random
#from quiver_engine import server
from pylab import *
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression
from skimage.transform import pyramid_gaussian
from keras.preprocessing.image import img_to_array

def image_resize(image, size=(27, 27)):       
    return cv2.resize(image, size) #, interpolation = cv2.INTER_AREA)


def pyramids(image, scale=1.5, minSize=(27, 27)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = cv2.resize(image,(w,w))
        if image.shape[0] < minSize[0] or image.shape < minSize[1]:
            break
        yield image


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

(winW, winH) = (27, 27)
'''
directory = 'C:/Users/макс/Desktop/rh/Desktop/image' 
files = os.listdir(directory)
labels = []
name = []
rawImages = []
for imagePath in files:
    image = cv2.imread(directory+'/'+imagePath,0)
    label = imagePath.split(os.path.sep)[-1].split(".")[1]
   
    kernel = np.ones((8,8),np.float32)/25
    image = cv2.filter2D(image,-1,kernel)
   
    rawImages.append(img_to_array(image))
    labels.append(label)
x_train = np.asarray(rawImages)
x_test = np.asarray(labels)
dataset_size = len(rawImages)
#x_train = rawImages.reshape(dataset_size,-1)
dataset_size = len(labels)
#x_test = labels.reshape(dataset_size,-1)

'''


input_shape = (1,27,27)

image2 = cv2.imread('11.jpg',0)


'''
brightness = 50
contrast = 30
img = np.int16(image)
img = img * (contrast/127+1) - contrast + brightness
img = np.clip(img, 0, 255)
image = np.uint8(img)
'''
output_y=np.array([1.0])
n_iter = 56
losses=[]

json_file = open('cifar10_architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
net = model_from_json(loaded_model_json)

# load weights into new model
net.load_weights("cifar10_weights.h5")
print("Loaded model from disk")

'''

def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

layer_outputs = [layer.output for layer in net.layers]
activation_model = Model(input=net.input, output=layer_outputs)
'''


scale=0
downscale=1.5
ins=0
detections = []

windowSize=(winW,winH)
scale=0
for (x, y, window) in sliding_window(image2, stepSize=10, windowSize=(winW, winH)):        
    if window.shape[0] != winH or window.shape[1] != winW:
        continue

    cv2.GaussianBlur(window,(5,5),0)
    window=img_to_array(window)/255   
   
    window = np.expand_dims(window, axis=0)
    n=net.predict_on_batch(window)[0]
    #print(n)
    if n[1]>0.97:                      
        ins+=1
        print("Detection:: Location -> ({}, {})".format(x, y))
        print(n)        
        detections.append((int(x * (downscale**scale)),int(y*(downscale**scale)),
                                  np.array([n[1]]),int(windowSize[0]*(downscale**scale)),
                                   int(windowSize[1]*(downscale**scale))) )
                
  
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) # do nms on the detected bounding boxes

sc = [score[0] for (x, y, score, w, h) in detections]
print("detection confidence score: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
N=0
for (xA, yA, xB, yB) in pick:
    #cv2.rectangle(image2, (xA, yA), (xB, yB), (205,205,20), 2)   
    #cv2.imshow("Raw Detections after NMS", image2[yA:yB,xA:xB])
    '''для вырезки изображений'''
    #cv2.imwrite("negative.2."+str(N)+".png", image2[yA:yB,xA:xB])   
    N+=1


for (xA, yA, xB, yB) in pick:
    cv2.rectangle(image2, (xA, yA), (xB, yB), (205,205,20), 2)   
    #cv2.imshow("Raw Detections after NMS", image2[yA:yB,xA:xB])     




#### Save the images below
k = cv2.waitKey(0) & 0xFF 
if k == 27:             #wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('of_saved_image.png',image2)
    cv2.destroyAllWindows()




        
cv2.imwrite("frame1.jpg", image2)        
            
    
cv2.destroyAllWindows()
