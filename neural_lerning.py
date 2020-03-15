# -*- coding: cp1251 -*-
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras import regularizers

import keras
import numpy as np
import cv2
import os
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



print(keras.__version__)


#from quiver_engine import server
# CIFAR_10 is a set of 60K images 32x32 pixels on 3 channels
IMG_CHANNELS = 1
IMG_ROWS = 27
IMG_COLS = 27

#constant
BATCH_SIZE = 128
NB_EPOCH = 60
NB_CLASSES = 2
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

INPUT_SHAPE = (27,27,1)


print("[INFO] loading images...")
directory = 'C:/Users/maksim/Desktop/image' 
files = os.listdir(directory)
labels = []
data = []
for imagePath in files:
    image = cv2.imread(directory+'/'+imagePath,0)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    #print(label)
    data.append(img_to_array(image))
    '''
    if image.shape[0]!=27 or  image.shape[1]!=27:
        print imagePath
    '''
    if label=="negative":
        labels.append(0)
    else:
        labels.append(1)
        
'''    
x_train = np.asarray(data)
x_test = np.asarray(labels)
dataset_size = len(rawImages)

dataset_size = len(labels)
'''
print( len(data))

data = np.asarray(data, dtype="float") / 255.0
labels = np.asarray(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))
 


(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.10,random_state=20)
print('X_train shape:', trainX.shape)
print(data.shape[0], 'train samples')
print(labels.shape[0], 'test samples')


# convert to categorical
trainY = np_utils.to_categorical(trainY, NB_CLASSES)
testY = np_utils.to_categorical(testY, NB_CLASSES)



callbacks_list = [ keras.callbacks.EarlyStopping( monitor='val_acc', patience=1, ),
                             keras.callbacks.ModelCheckpoint( filepath='my_model.h5',
                                 monitor='val_loss', save_best_only=True, )]


# network

def seg_model(n_classes):
    model = Sequential()
    model.add(Conv2D(16,( 3,3),bias=True,
		 border_mode='same',
		 activation='relu',
		init='glorot_uniform',  W_regularizer=regularizers.l2(0.01),
		input_shape=INPUT_SHAPE))
    model.add(Conv2D(16, (3,3),bias=True,
		 border_mode='same',
		 activation='relu',
		init='glorot_uniform',  W_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(32, (3,3),bias=True,
		 border_mode='same',
		 activation='relu',
		init='glorot_uniform',  W_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(32,( 3,3),bias=True,
		 border_mode='same',
		 activation='relu',
		init='glorot_uniform',  W_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
	
    #Fully connected layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu', W_regularizer=regularizers.l2(0.01)))
    model.add(Dense(64, activation='relu', W_regularizer=regularizers.l2(0.01)))
    model.add(Dense(n_classes, activation='softmax', W_regularizer=regularizers.l2(0.01)))
    return model


model=seg_model(2)


model.summary()

# train
optim = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optim,
	metrics=['accuracy'])


'''

x_train = x_train.reshape((dataset_size, 10000,-1))
'''

history = model.fit(trainX, trainY, batch_size=BATCH_SIZE,nb_epoch=NB_EPOCH,
	callbacks=callbacks_list, 
	verbose=VERBOSE)
 


print('Testing...')

score = model.evaluate(testX, testY,
                     batch_size=BATCH_SIZE, verbose=VERBOSE)
'''
score = model.evaluate(x_train, x_test,
                     batch_size=BATCH_SIZE, verbose=VERBOSE)
'''                    
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

#server.launch(model)


#save model
model_json = model.to_json()
open('architecture.json', 'w').write(model_json)
model.save_weights('weights.h5', overwrite=True)


plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

