

from __future__ import division
import numpy as np
import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from imageio import imread as imread
from imageio import imwrite as imsave
import sys
sys.path.append('./script')
from fprocess import mcpredNall,mcpredNauxi
from fmisc import CustomCallbackkernel
from fmisc import CustomCallback, CustomCallbackpretrain, CustomCallbackmrf
from fmisc import myloss, mylosssparse, configdims, passweights

from finteract import getChainlist,getLabelSpSynthetic,getLabelSpReal
from finteract import getConfiMask, InitialPred2Sp
from finteract import propAnnoMapsyn, propAnnoMapreal

from finteract import imshow2, clickXY, collectLabel

from fmisc import getWcls
from fmisc import pltshow, pltshow2
from fmisc import ioulogitIndividual as ioulogit
from fsoften import calculateSlic

import pickle
from matplotlib import pyplot as plt
import cv2


from fmisc import getWcls


from fmisc import N,Nbatch,dimdata,dimcls,h1,w1
from fmisc import npat,priors,priors2,N2,Nsample

dims = configdims(dimdata, dimcls, 20,40,40)
realuser = 1 
nclicks = 10



adamverylarge = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adamlarge = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adamsmall = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adamverysmall = tf.keras.optimizers.Adam(learning_rate=0.00025, beta_1=0.9, beta_2=0.99, epsilon=1e-06,)
sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
sgd0step = tf.keras.optimizers.SGD(learning_rate=0.)
verbose = 0


x = Input(batch_shape=(Nbatch, h1,w1, dimdata), name='inputx')

kldiv = N*h1*w1
kldiv2 = Nbatch*h1*w1
kldivtrial = 100*h1*w1
kldivs = [kldivtrial, kldiv2]



auxi = np.zeros((N,1)).astype('float32')
for i in range(1,int(N/Nbatch)):
    auxi[i*Nbatch:(i+1)*Nbatch] = i*Nbatch
xaux = Input(batch_shape=(Nbatch, 1), name='inputaux')

from fmodel import adaZpretrain as myNNpretrain
params = [N,Nbatch,npat,1e-5]

trn10 = []
for i in range(npat):
    trn10.append(True)
trainables = [True,True, trn10, True, True]

m = Model(inputs=[x,xaux], 
          outputs=myNNpretrain([x,xaux], dims, params, priors, priors2, trainables, kldivs))
m.compile(loss={'lnow':mylosssparse, 
                        },
          loss_weights={'lnow': 1.0,},
          optimizer=adam, 
          metrics={'lnow':'accuracy',})







from fprocess import initializezdata,initializez
from fprocess import getlayeridx,getlayerweights


m = initializezdata(m)
m = initializez(m)
m0weights = m.get_weights()

mode = 1
if mode==1:
    m.load_weights('script/0.h5')
    m0weights = m.get_weights()


from fmisc import mylossent
from fmodel import adaZshare as myNNtest

Nbatch2 = 1
params2 = [Nbatch2,Nbatch2,npat,1e-5]
trn102 = []
for i in range(npat):
    trn102.append(False)
trainables2 = [False,False, trn102, True, False]
auxi2 = np.zeros((Nbatch2,1)).astype('float32')
xaux2 = Input(batch_shape=(Nbatch2, 1), name='inputaux')





x2 = Input(batch_shape=(Nbatch2, h1,w1, dimdata), name='inputx')
m2 = Model(inputs=[x2,xaux2], 
          outputs=myNNpretrain([x2,xaux2], dims, params2, priors, priors2, trainables2, kldivs))
from fprocess import passweightsm2
idxZall = 66 
m2 = passweightsm2(m,m2,idxZall)

m2.compile(loss={'lnow':mylossent,
                        },
          loss_weights={'lnow': 1.0,},
          optimizer=adam, 
          metrics={'lnow':'accuracy',})




params2s = [Nsample,Nsample,npat,1e-5]

auxi2s = np.zeros((Nsample,1)).astype('float32')
xaux2s = Input(batch_shape=(Nsample, 1), name='inputaux')

x2s = Input(batch_shape=(Nsample, h1,w1, dimdata), name='inputx')
m2s = Model(inputs=[x2s,xaux2s], 
          outputs=myNNtest([x2s,xaux2s], dims, params2s, priors, priors2, trainables2, kldivs))
m2s.compile(loss={'lnow':mylossent,
                        },
          loss_weights={'lnow': 1.0,},
          optimizer=adam, 
          metrics={'lnow':'accuracy',})





prior1names = []
for i in range(1,8):
    prior1names.append('b'+str(i)+'a')
    prior1names.append('b'+str(i)+'b')
prior1names.append('lnow')
prior1names.append('f2a')

prior2names = []
for i in range(5,8):
    for j in range(npat):
        prior2names.append('b'+str(i)+'m'+str(j))





from fmisc import getTestData
filepath = 'script/imgs.pkl'

mode = 1
if mode==1:
    images2, labels2, labelsnum2, testingfiles = pickle.load(open(filepath,"rb"))

wcls = getWcls(labelsnum2, dimcls)

for idxseq in range(1):
    
    trndata = images2[idxseq:idxseq+1,]
    trnlabel = labelsnum2[idxseq:idxseq+1,].astype(np.int32)
    trnlabel1hot = labels2[idxseq:idxseq+1,]
    clstruth = trnlabel[0,:]
    
    fileimg = 'output/'+str(idxseq)+'img.png'
    pltshow(trndata[0],fileimg)
    filename = 'output/'+str(idxseq)+'gt.png'
    pltshow(trnlabel[0],filename)
    
    
    trndatas = np.tile(trndata, [Nsample,1,1,1])
    trnlabels = np.tile(trnlabel, [Nsample,1,1])
    
    m2s = passweights(m2,m2s)
    m2s.fit([trndatas,auxi2s],
          {'lnow': trnlabels,
           },
          batch_size=Nsample,
          epochs=100,
          verbose=0,)
    
    m2 = passweights(m2s,m2)
    
    meanpred, clspred, uncertain = mcpredNauxi(m2,trndata,auxi2, dimcls, 32)
    filename = 'output/'+str(idxseq)+'initial.png'
    pltshow(clspred[0],filename)
    filename = 'output/'+str(idxseq)+'ent.png'
    pltshow2(uncertain[0],filename)
    result = ioulogit(to_categorical(clstruth,dimcls), to_categorical(clspred[0],dimcls),\
                         wcls, idxseq, 'initialpred')
    
    
    from fprocess import preparePriors3
    priors1b,priors2b = preparePriors3(m2,0,0,prior1names,prior2names)
    
    newimage = cv2.imread(fileimg, cv2.IMREAD_UNCHANGED)
    imgpath = "script/"+str(idxseq)+".jpg"
    filepath = "script/"+str(idxseq)+".pkl"
    cv2.imwrite(imgpath, newimage)
    
    mode = 0
    if mode==0:
        num_sp,mask_sp,size_sp,center_sp,lab_sp,hist_sp,hist_sp2 \
            = calculateSlic(imgpath)
        pickle.dump([num_sp,mask_sp,size_sp,center_sp,lab_sp,hist_sp,hist_sp2], \
                    open(filepath, "wb"))
    else:
        num_sp,mask_sp,size_sp,center_sp,lab_sp,hist_sp,hist_sp2 \
            = pickle.load(open(filepath,"rb"))
            
    
    initialpredSp = InitialPred2Sp(meanpred[0], mask_sp, size_sp)
    confimask = getConfiMask(uncertain[0], 1)
    
    pltshow(initialpredSp)
    pltshow2(confimask)
    
    chainlist = getChainlist(num_sp, center_sp, hist_sp,hist_sp2)
    
    from finteract import getNearlist
    nearlist = getNearlist(num_sp, center_sp)
    from finteract import portionSp, criteriaSp
    label_sp, prop_sp = portionSp(mask_sp,size_sp,trnlabel1hot)
    criteria_sp = criteriaSp(chainlist, nearlist, label_sp, prop_sp, center_sp, h1,w1)
    
    
    
    
    
    anno_sp = np.zeros((num_sp))
    skip_sp = np.zeros((num_sp))
    clickpos = []
    clicklabel = []

    
    correct = (clstruth==clspred[0])
    acc_sp = np.zeros((num_sp))
    confi_sp = np.zeros((num_sp))
    
    for i in range(num_sp):
        temp = np.sum(correct * mask_sp[i,:]) / size_sp[i]
        acc_sp[i] = temp
        temp2 = np.sum(confimask * mask_sp[i,:]) / size_sp[i]
        confi_sp[i] = temp2
    
    for usriter in range(1):
        if realuser==0: 
            countnow = 0
            threshold = 32**2
            accthresh =0.25
            tempacc = np.copy(acc_sp)
            tempacc[skip_sp==1] = 4
            tempacc[acc_sp>accthresh] = 3
            
            tempacc[criteria_sp==False] = 5
            
            accmarked = np.zeros((num_sp))
            for i in range(nclicks):
                idxnow = np.argmin(tempacc)
                if tempacc[idxnow]<1:
                    anno_sp[idxnow] = 1
                    countnow +=1
                    for j in range(num_sp):
                        distsquare = np.sum(np.square(center_sp[idxnow]-center_sp[j]))
                        if distsquare<threshold:
                            accmarked[j] = 1
                        if distsquare<threshold and tempacc[j]<=1:
                            tempacc[j] = 2 
                    tempacc[idxnow] = 4
            
            if countnow < nclicks:
                tempconfi = np.copy(confi_sp)
                tempconfi[skip_sp==1] = 4                    
                tempconfi[criteria_sp==False] = 5
                
                tempconfi[anno_sp==1] = 4
                tempconfi[accmarked==1] = 3 
                for i in range(countnow, nclicks):
                    idxnow = np.argmin(tempconfi)
                    anno_sp[idxnow] = 1
                    countnow +=1
                    for j in range(num_sp):
                        distsquare = np.sum(np.square(center_sp[idxnow]-center_sp[j]))
                        if distsquare<threshold and tempconfi[j]<=1:
                            tempconfi[j] += 1
                    tempconfi[idxnow] = 4
            
            
            mode = 0
            if mode==0:
                spmap1 = np.zeros((h1,w1))
                spacc1 = np.zeros((h1,w1))
                for i in range(num_sp):
                    spacc1 += mask_sp[i]*(1-acc_sp[i])
                    if anno_sp[i]==1:                        
                        spmap1 += mask_sp[i]*label_sp[i]
                
        elif realuser==1: 
            
            imshow2(clspred[0], trndata[0], dimcls, h1, w1)
            clickpos = clickXY(2*h1, w1, nclicks, clickpos)
            clicklabel = collectLabel(nclicks, clicklabel)
            label_sp,anno_sp,skip_sp,click_sp \
                = getLabelSpReal(mask_sp,clickpos,clicklabel,anno_sp,skip_sp)
    
    
       
        if realuser==0: 
            mask_usr,labelmap1 \
                = propAnnoMapsyn(clspred[0],chainlist,center_sp,anno_sp,mask_sp,label_sp,2)
        elif realuser==1: 
            mask_usr,labelmap1 \
                = propAnnoMapreal(clspred[0], clickpos,clicklabel, chainlist,anno_sp,mask_sp,2)  
    
    
    
        
        
        from skimage import color
        from fsoften import pixelRGBsimilarity, mprop
        from fsoften import softlabelmap, initialweightsprop
        
        labarr = color.rgb2lab(trndata[0])/100
        similaritys = pixelRGBsimilarity(labarr, 0.1)
        
        
        
        mask_usr_weighted = np.copy(mask_usr)
        mask_usr_weighted += 128*(mask_usr==2)
        mask_usr_weighted += 1*(mask_usr==1)
        
        mask_usr_weighted += 1*(0.5*confimask+0.5)*(mask_usr==0)
        
        mpropmask = np.expand_dims(mask_usr_weighted, axis=0)
        
        
        softmap = softlabelmap(meanpred[0], to_categorical(labelmap1, dimcls), mask_usr)
        
        
        xmprop = Input(batch_shape=(1,))
        paramsmrf = [h1,w1,dimcls, similaritys, 0.5]
        mmprop = Model(inputs=xmprop,
                  outputs=mprop(xmprop,paramsmrf))
        
        mmprop.compile(loss={'lnow': 'categorical_crossentropy',}, 
                  loss_weights={'lnow': 1.,},
                  optimizer=adamverylarge,
                  metrics={'lnow':'accuracy',})   
        
        mmprop = initialweightsprop(mmprop, softmap, 1e-2, 0.5)
        
        mpropdummy = np.zeros((1,))
        mmprop.fit(mpropdummy,
              {'lnow': np.expand_dims(softmap, axis=0)},
              batch_size=1,
              sample_weight = mpropmask,
              epochs=301,
              verbose=0,)
        
        propraw = mmprop.get_weights()[0]
        prophard = propraw.argmax(-1)
        propsoft = propraw - np.max(propraw, axis=-1, keepdims=True)
        propsoft = np.exp(propsoft)/np.sum(np.exp(propsoft), axis=-1, keepdims=True)

    


    trainables3 = [False,False, trn10, True, True]

    kldivs3 = [kldivtrial, h1*w1/4]
    
    m3 = Model(inputs=[x2,xaux2], 
              outputs=myNNtest([x2,xaux2], dims, params2, priors1b, priors2b, trainables3, kldivs3))
    m3.compile(loss={'lnow':mylosssparse,
                            },
              loss_weights={'lnow': 1.0,},
              optimizer=adam, 
              metrics={'lnow':'accuracy',})
    
    m3.set_weights(m2s.get_weights())

    m3mask = np.clip(mpropmask,0,10)
    
    m3.compile(loss={'lnow':myloss, 
                        },
          loss_weights={'lnow': 1.0,},
          optimizer = adamsmall, 
          metrics={'lnow':'accuracy',})
    target = np.expand_dims(propsoft, axis=0)
    
    mode = 0
    if mode==0: 
        m3.fit([trndata,auxi2],
              {'lnow': target, 
               },
              epochs=501,
              sample_weight = m3mask,
              verbose=verbose,)
    
    m2 = passweights(m3,m2)
    
    meanpred, clspred, uncertain = mcpredNauxi(m2,trndata, auxi2, dimcls, 50)


    postprocess = 1
    if postprocess==1:
        retrainedSuper = InitialPred2Sp(meanpred[0], mask_sp, size_sp)
        mode = 12
        if mode==12: 
            _,labelmap2 \
                = propAnnoMapsyn(clspred[0],chainlist,center_sp,anno_sp,mask_sp,label_sp,2)
        elif mode==13: 
            _,labelmap2 \
                = propAnnoMapreal(clspred[0], clickpos,clicklabel, chainlist,anno_sp,mask_sp,2)

        
        softmap = softlabelmap(meanpred[0], to_categorical(labelmap2, dimcls), mask_usr)
        mmprop = Model(inputs=xmprop,
                  outputs=mprop(xmprop,paramsmrf))
        mmprop.compile(loss={'lnow': 'categorical_crossentropy',}, 
                  loss_weights={'lnow': 1.,},
                  optimizer=adamverylarge,
                  metrics={'lnow':'accuracy',})   
        mmprop = initialweightsprop(mmprop, softmap, 1e-2, 2)
        mmprop.fit(mpropdummy,
              {'lnow': np.expand_dims(softmap, axis=0)},
              batch_size=1,
              sample_weight = mpropmask,
              epochs=401,
              verbose=0,
              callbacks=[CustomCallbackmrf()])
        propraw = mmprop.get_weights()[0]
        
        prophard = propraw.argmax(-1)

        filename = 'output/'+str(idxseq)+'-refine.png'
        pltshow(prophard)
        
        result = ioulogit(to_categorical(clstruth,dimcls), to_categorical(prophard,dimcls),\
                             wcls, idxseq, 'refine')
    
        
