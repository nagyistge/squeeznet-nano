
#Nano material shaper prediction using squeezenet
#Author: Gananath R
#https://github.com/Gananath 

#I am thankful to thie github repo https://github.com/rcmalli/keras-squeezenet
#Squeenet weight and codes were taken from this repo and modified it for this project
#if you want to learn more about it then please visit the link

from scipy import misc
import copy
import os,random
import numpy as np
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dropout, Activation, Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop
from time import sleep

#number of episodes
episodes=1

#folder path
path="/home/gananath/Desktop/squeezenet-nano"



sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"


#original fire_modulce code by https://github.com/rcmalli
def fire_module(x, fire_id, squeeze=16, expand=64, dim_ordering='th'):
    s_id = 'fire' + str(fire_id) + '/'
    if dim_ordering is 'tf':
        c_axis = 3
    else:
        c_axis = 1

    x = Convolution2D(squeeze, 1, 1, border_mode='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, 1, 1, border_mode='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, 3, 3, border_mode='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = merge([left, right], mode='concat', concat_axis=c_axis, name=s_id + 'concat')
    return x

#modified squeezenet model for predicting shape of the nanoparticle from image
def nanonet(nb_classes=1000, dim_ordering='th'):
    img_input = Input(shape=(3, 227, 227))
    nn_layer = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid', name='conv1')(img_input)
    nn_layer = Activation('relu', name='relu_conv1')(nn_layer)
    nn_layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(nn_layer)

    nn_layer = fire_module(nn_layer, fire_id=2, squeeze=16, expand=64, dim_ordering=dim_ordering)
    nn_layer = fire_module(nn_layer, fire_id=3, squeeze=16, expand=64, dim_ordering=dim_ordering)
    nn_layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(nn_layer)

    nn_layer = fire_module(nn_layer, fire_id=4, squeeze=32, expand=128, dim_ordering=dim_ordering)
    nn_layer = fire_module(nn_layer, fire_id=5, squeeze=32, expand=128, dim_ordering=dim_ordering)
    nn_layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(nn_layer)

    nn_layer = fire_module(nn_layer, fire_id=6, squeeze=48, expand=192, dim_ordering=dim_ordering)
    nn_layer = fire_module(nn_layer, fire_id=7, squeeze=48, expand=192, dim_ordering=dim_ordering)
    nn_layer = fire_module(nn_layer, fire_id=8, squeeze=64, expand=256, dim_ordering=dim_ordering)
    nn_layer = fire_module(nn_layer, fire_id=9, squeeze=64, expand=256, dim_ordering=dim_ordering)
    nn_layer = Dropout(0.5, name='drop9')(nn_layer)

    nn_layer = Convolution2D(nb_classes, 1, 1, border_mode='valid', name='conv10')(nn_layer)
    nn_layer = Activation('relu', name='relu_conv10')(nn_layer)
    nn_layer = GlobalAveragePooling2D()(nn_layer)
    nn_layer = Activation('tanh')(nn_layer)
    nn_layer = Dense(20,activation='relu')(nn_layer)
    nn_layer = Dropout(0.3)(nn_layer)
    nn_layer = Dense(3)(nn_layer)
    out_layer = Activation('softmax', name='loss')(nn_layer)
    model = Model(input=img_input, output=[out_layer])
    return model

#Image pre-processing before feeding it to neural net    
def image_proc(fpath):
    im = misc.imread(fpath)
    im = misc.imresize(im, (227, 227)).astype(np.float32)
    aux = copy.copy(im)
    im[:, :, 0] = aux[:, :, 2]
    im[:, :, 2] = aux[:, :, 0]
    
    # Remove image mean
    im[:, :, 0] -= im[:, :, 0].mean()
    im[:, :, 1] -= im[:, :, 1].mean()
    im[:, :, 2] -= im[:, :, 2].mean()
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return im
     

#training model from folder images
def train_model(model,path):
    for i in range(0,episodes):
        np.random.seed(random.randint(1,100))
        #randomizing folder order
        folder=np.random.permutation(os.listdir(path))
        for dirc in folder:
		for files in os.listdir(path+dirc):
			if dirc=='Carbon nanotubes':
				y=np.array([[1,0,0]])
			elif dirc=='Particles':
				y=np.array([[0,1,0]])
			else:
				y=np.array([[0,0,1]])
			print dirc
			print files
			print i
			X=image_proc(path+dirc+"/"+files)				
			model.fit(X,y,batch_size=1, nb_epoch=10)
			sleep(5)
    return model
    
 
model = nanonet()
#rmsprop=RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])
model.load_weights(path+'/squeezenet-nanoWts.h5', by_name=True)

#training model
train_model(model,path+'/Datasets/')

#saving new model       
model.save(path+'/squeezenet-nanoWts.h5')        

im=image_proc(path+'/Datasets/Carbon nanotubes/0002.jpg')
#im=image_proc(path+'/Datasets/None/416SL-7R1QL.jpg')
#im=image_proc(path+'/Datasets/Particles/15a.jpg')
np.argmax(model.predict(im))
