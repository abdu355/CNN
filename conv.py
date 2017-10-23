# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:02:41 2017

@author: GIGABYTE
"""
# data already seperated into directories for easy import
# -------------Part 1 - building CNN---------------
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=opencl0:0,floatX=float32,gpuarray.preallocate=-1"
from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
    
os.environ['KERAS_BACKEND']='tensorflow'
#os.environ['KERAS_BACKEND']='theano'


from timeit import default_timer as timer
start = timer()

from keras.models import Sequential #initialize NN
from keras.layers import Conv2D # add convolution layers from 2D images (videos are 3D with time)
from keras.layers import MaxPooling2D # pooling layer from 2D conv layers
from keras.layers import Flatten # for ANN input (fully connected layers)
from keras.layers import Dense # add the fully connected layers to the ANN


#initialize CNN

classifier = Sequential()

# Step 1 - Convolution

#filters: no. of filters (feature detectors) that will generate one feature map (32 detectors, kernel_size)
#kernel_size: (rows , cols)
#input_shape: 3d array (colored) 3 channels / 2d array (b&W) 1 channel (rows, cols, channels) tensorflow format
#activation: 'relu' rectifier function for non-linearity (to remove negative pixels and classify images correctly)
classifier.add(Conv2D(filters = 32, 
                      kernel_size = (3, 3),
                      activation = 'relu',
                      input_shape = (64, 64, 3)))
# Step 2 - Pooling

#pool_size: (2,2) => 2 by 2 matrix that will stride over the feature map and reduce its size to form a new pooled matrix
#this will divide the size of feature map by 2
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#2nd conv layer for additonal accuracy
classifier.add(Conv2D(32, 
                      (3, 3), 
                      activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flatenning

#this will flatten the last layer of the CNN to form the input for ANN
classifier.add(Flatten())

# Step 4 - Full Connection (Hidden Layer)

#input matrix => 32 feature maps => size reduced to half => still alot of nodes !
#choose power of 2 (large no. usually 128+)
classifier.add(Dense(128, activation = 'relu'))

#binary outcome (sigmoid) - more than 2 outcomes (softmax)
#1 output layer
classifier.add(Dense(1, activation = 'sigmoid')) 

#Compiling CNN
#optimizer: Stochastic Gradient (adam)
#loss: logarithmic loss (logistic regression) => binary_crossentropy else for more outcomes we use categorical_crossentropy
#metrics: look for accuracy
classifier.compile(optimizer = 'adam', 
                   loss= 'binary_crossentropy',
                   metrics = ['accuracy'])

# -------------Part 1 - Fitting CNN to Images ---------------

#Augment images to prevent overfitting with small amount of images
#ImageDataGenerator configuration goes here
#<boiler plate code from Keras Docs>
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#create training_set
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#create test_set
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#Fit the model
classifier.fit_generator(
        training_set,
        steps_per_epoch=250, #8000/32
        epochs=25, 
        validation_data=test_set,
        validation_steps=63) #2000/32
# elapsed time
end = timer()
print(end - start)
# end of work message
import os
os.system('say "Training Complete"')

#---------- visualizing the convolution of images---------

import cv2
import numpy as np
import matplotlib.pyplot as plt
cat = cv2.imread('cat4001.jpg')
print(cat)
#cv2.imshow('image', cat)
#plt.imshow(cat)

classifier = Sequential()

# Step 1 - Convolution

#filters: no. of filters (feature detectors) that will generate one feature map (32 detectors, kernel_size)
#kernel_size: (rows , cols)
#input_shape: 3d array (colored) 3 channels / 2d array (b&W) 1 channel (rows, cols, channels) tensorflow format
#activation: 'relu' rectifier function for non-linearity (to remove negative pixels and classify images correctly)
classifier.add(Conv2D(filters = 3, 
                      kernel_size = (3, 3),
                      input_shape = cat.shape))

cat_batch = np.expand_dims(cat,axis=0)
conv_cat = classifier.predict(cat_batch)

def visualize_cat(cat_batch):
    cat = np.squeeze(cat_batch, axis=0)
    print (cat.shape)
    plt.imshow(cat)
    
visualize_cat(conv_cat)

classifier = Sequential()
classifier.add(Conv2D(filters = 1, 
                      kernel_size = (3, 3),
                      activation = 'relu',
                      input_shape = cat.shape))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(filters = 1, 
                      kernel_size = (3, 3),
                      activation = 'relu',
                      input_shape = cat.shape))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(filters = 1, 
                      kernel_size = (3, 3),
                      activation = 'relu',
                      input_shape = cat.shape))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


def nice_cat_printer(model, cat):
    cat_batch = np.expand_dims(cat, axis = 0)
    conv_cat2 = classifier.predict(cat_batch)
    
    conv_cat2 = np.squeeze(conv_cat2, axis = 0)
    print (conv_cat2.shape)
    conv_cat2 = conv_cat2.reshape(conv_cat2.shape[:2])
    
    print(conv_cat2.shape)
    plt.imshow(conv_cat2)


nice_cat_printer(classifier, cat)
#1 conv layer results: (~17 min) - CPU only
#0.3390 - acc: 0.8493 - val_loss: 0.5134 - val_acc: 0.7890
#training_set : 85%
#test_set: 79%


#2 conv layer results: (1045 seconds) - CPU Only
#0.3146 - acc: 0.8645 - val_loss: 0.4366 - val_acc: 0.8085
#training_set :  86.5%
#test_set: 81%

#2 conv layer results: (X seconds) - GPU Only (Theano)
#
#training_set : 
#test_set: 