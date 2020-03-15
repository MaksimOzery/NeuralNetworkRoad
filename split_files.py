# -*- coding: cp1251 -*-

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
(winW, winH) = (27, 27)

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

directory = r'C:\Users\maksim\Desktop\1' 
files = os.listdir(directory)
windowSize=(winW,winH)
n=0
p=0
for imagePath in files:
    n=0
    image = cv2.imread(directory+'/'+imagePath,0)
    for (x, y, window) in sliding_window(image, stepSize=10, windowSize=(winW, winH)):        
        if window.shape[0] != winH or window.shape[1] != winW:
            continue                
        cv2.imwrite("C:/Users/maksim/Desktop/7/"+str(n)+"."+str(p)+".jpg", window)        
        print( n)
        n+=1
    p+=1
cv2.destroyAllWindows()
