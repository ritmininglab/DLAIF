import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import os

from imageio import imread as imread
from imageio import imwrite as imsave
from skimage.transform import resize as imresize

from matplotlib import pyplot as plt

def pltshow(data,filename=0):
    if filename==0:
        plt.figure()
        plt.imshow(data)
        plt.axis('off')
        plt.show() 
    else:
        plt.figure()
        plt.imshow(data)
        plt.axis('off')
        plt.savefig(filename,bbox_inches='tight')
        plt.show() 
def pltshow2(data,filename=0):
    if filename==0:
        plt.figure()
        plt.imshow(data, cmap='coolwarm', vmin=-.75, vmax=1.25)
        plt.axis('off')
        plt.show() 
    else:
        plt.figure()
        plt.imshow(data, cmap='coolwarm', vmin=-.75, vmax=1.25)
        plt.axis('off')
        plt.savefig(filename,bbox_inches='tight')
        plt.show() 

def getTrainData(N,h1,w1,dimclass):
    train_data = []
    datatrain_path = 'train/'
    
    training_images = [datatrain_path + f for f in os.listdir(datatrain_path) if f.endswith(('.jpg', '.jpeg'))] 
    training_images.sort()
    for image in training_images:
        annotation_data = {'image': image}
        train_data.append(annotation_data) 
    
    images = np.zeros((N, h1,w1, 3), dtype=np.float32)
    for ii, data in enumerate(train_data):
        r_img = imread(data['image'])
        images[ii, :] = np.copy(r_img / 255)
    
    labels = np.zeros((N, h1, w1, dimclass))
    labelsnum = np.zeros((N, h1, w1), dtype=np.int8)
    labelpath = [datatrain_path + f for f in os.listdir(datatrain_path) if f.endswith(('.csv'))] 
    labelpath.sort()
    for ii, lb in enumerate(labelpath):
        y1 = np.loadtxt(open(lb, "rb"), delimiter=",")
        labelsnum[ii, :] = np.copy(y1)
        y1 = to_categorical(y1,dimclass)
        labels[ii, :] = np.copy(y1)
        
    labels = labels.astype(np.int8)
    
    return images, labels, labelsnum, training_images

def getTestData(N2,h1,w1,dimclass):

    test_data = []
    datatest_path = 'test/'
    
    testing_images = [datatest_path + f for f in os.listdir(datatest_path) if f.endswith(('.jpg', '.jpeg'))] 
    testing_images.sort()
    for image in testing_images:
        annotation_data = {'image': image}
        test_data.append(annotation_data) 
    
    images2 = np.zeros((N2, h1,w1, 3), dtype=np.float32)
    for ii, data in enumerate(test_data):
        img = imread(data['image'])    
        images2[ii, :] = np.copy(img / 255)
    
    labels2 = np.zeros((N2, h1, w1, dimclass))
    labelsnum2 = np.zeros((N2, h1, w1), dtype=np.int8)
    labelpath2 = [datatest_path + f for f in os.listdir(datatest_path) if f.endswith(('.csv'))] 
    labelpath2.sort()
    for ii, lb in enumerate(labelpath2):
        y1 = np.loadtxt(open(lb, "rb"), delimiter=",")
        labelsnum2[ii, :] = np.copy(y1)
        y1 = to_categorical(y1,dimclass)
        labels2[ii, :] = np.copy(y1)
    
    labels2 = labels2.astype(np.int8)
    
    return images2, labels2, labelsnum2, testing_images

def getCoreData(N,h1,w1,dimclass):
    train_data = []
    datatrain_path = 'coreset/'
    
    training_images = [datatrain_path + f for f in os.listdir(datatrain_path) if f.endswith(('.jpg', '.jpeg'))] 
    training_images.sort()
    for image in training_images:
        annotation_data = {'image': image}
        train_data.append(annotation_data) 
    
    images = np.zeros((N+1, h1,w1, 3), dtype=np.float32)
    for ii, data in enumerate(train_data):
        r_img = imread(data['image'])
        images[ii, :] = np.copy(r_img / 255)
    
    labels = np.zeros((N+1, h1, w1, dimclass))
    labelsnum = np.zeros((N+1, h1, w1), dtype=np.int8)
    labelpath = [datatrain_path + f for f in os.listdir(datatrain_path) if f.endswith(('.csv'))] 
    labelpath.sort()
    for ii, lb in enumerate(labelpath):
        y1 = np.loadtxt(open(lb, "rb"), delimiter=",")
        labelsnum[ii, :] = np.copy(y1)
        y1 = to_categorical(y1,dimclass)
        labels[ii, :] = np.copy(y1)
        
    labels = labels.astype(np.int8)
    
    return images, labels, labelsnum, training_images



class CustomCallbackpretrain(Callback):
    def on_train_begin(self, logs={}):    
        self.epochs = 0    
    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % 10 != 11:     
            print("epoch {}, loss {:3.3f}={:3.3f}+{:3.3f} acc {:1.3f}".format(
                self.epochs, logs["loss"], logs["lnow_loss"], logs["loss"]-logs["lnow_loss"],
                logs["lnow_accuracy"])
                )
class CustomCallback(Callback):
    def on_train_begin(self, logs={}):    
        self.epochs = 0    
    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % 20 == 1:     
            print("epoch {}, loss {:3.3f}={:3.3f}+{:3.3f} acc {:1.3f}".format(
                self.epochs, logs["loss"], logs["lnow_loss"], logs["loss"]-logs["lnow_loss"],
                logs["lnow_accuracy"])
                )
class CustomCallbackkernel(Callback):
    def on_train_begin(self, logs={}):    
        self.epochs = 0    
    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % 10 == 1:     
            print("epoch {}, loss {:3.3f}".format(
                self.epochs, logs["loss"])
                )
class CustomCallbackmrf(Callback):
    def on_train_begin(self, logs={}):    
        self.epochs = 0    
    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % 100 == 1:     
            print("epoch {}, loss {:3.3f}={:3.3f}+{:3.3f} acc {:1.3f}".format(
                self.epochs, logs["loss"], logs["lnow_loss"], logs["loss"]-logs["lnow_loss"],
                logs["lnow_accuracy"])
                )

def mylossuseless(y_true, y_pred):
    temp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true,logits=y_pred))
    return temp
def myloss(y_true, y_pred):
    temp = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true,logits=y_pred)
    return temp
def mylosssparse(y_true, y_pred):
    temp = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true,logits=y_pred)
    return temp

def mylossent(y_true, y_pred):
    temp2 = tf.nn.softmax(y_pred)
    temp = -tf.multiply(temp2, tf.math.log(temp2+1e-8))
    return temp

N = 800 
Nbatch = 4
dimdata = 3
dimcls = 2
h1 = 256
w1 = 352
Nsample = 2
N2 = 8

priors = []
for idx in range(15):
    wpriormu = 0.
    wpriorvar = 1/2 
    bpriormu = 0.
    bpriorvar = 1/2 
    priors.append([wpriormu, wpriorvar, bpriormu, bpriorvar])
aprior = 2.
bprior = 1.
priors.append([aprior,bprior]) 


npat = 5
priors2 = []
for idx in range(3*npat):
    wpriormu = 0.
    wpriorvar = 1/2 
    bpriormu = 0.
    bpriorvar = 1/2 
    priors2.append([wpriormu, wpriorvar, bpriormu, bpriorvar, wpriormu, wpriorvar])

def configdims(dimdata, dimclass, d1,d2,d3):
    d4 = d2+d3
    dims = [[dimdata,d1,d2,d2,d2,d2,d2,d3, d4,d3,d4,d3,d4,d3,d3],
        [d1,d2,d2,d2,d2,d2,d3,d3,d3,d3,d3,d3,d3,d3,dimclass]]
    return dims

def passweights(mfrom, mto):
    weights = mfrom.get_weights()
    mto.set_weights(weights)
    return mto

def getWcls(clstruth, dimcls):
    result = np.zeros((dimcls,))
    for cc in range(dimcls):
        result[cc] = np.sum(clstruth==cc)
    result = result/np.sum(result)
    return result

def ioulogitAggregate(clstruth, clspred, wcls, taskid=999, string='unknown'):
    ypred = np.argmax(clspred, axis=-1)
    ytrue = np.argmax(clstruth, axis=-1)
    iou = 0
    for cc in range(0,wcls.shape[0]):
        intersect = np.sum((ypred==cc)*(ytrue==cc))
        union = np.sum(((ypred==cc)+(ytrue==cc)) > 0)
        if union>0:
            iou += intersect / union * wcls[cc]
    print("case",taskid,string," iou = ", iou)
    
    return iou

def ioulogitIndividual(clstruth, clspred, wcls, taskid=999, string='unknown'):
    ypred = np.argmax(clspred, axis=-1)
    ytrue = np.argmax(clstruth, axis=-1)
    iou = 0
    for cc in range(0,wcls.shape[0]):
        intersect = np.sum((ypred==cc)*(ytrue==cc))
        union = np.sum(((ypred==cc)+(ytrue==cc)) > 0)
        if union>0:
            iou += intersect / union * np.sum(ytrue==cc)
    iou = iou / (ytrue.shape[-1]*ytrue.shape[-2])
    print("case",taskid,string," iou = ", iou)
    
    return iou

def npsigmoid(x):
    y = 1/(1 + np.exp(-x)) 
    return y
def npdesigmoid(x, clipvalue):
    xclip = np.clip(x, clipvalue, 1-clipvalue)
    y = np.log(xclip/(1-xclip))
    return y
def clipsigmoid(x, clipthreshold):
    xclip = np.clip(x, -clipthreshold, clipthreshold)
    y = 1/(1 + np.exp(-xclip)) 
    return y    
