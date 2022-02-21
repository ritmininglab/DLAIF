
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers

def getlayeridx(model, layerName):
    index = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layerName:
            index = idx
            break
    return index
def getlayerweights(model, layername):
    idx = getlayeridx(model, layername)
    return model.layers[idx].get_weights() 


def initializezdata(m):
    idx1 = getlayeridx(m, 'f1a')
    abweights = m.layers[idx1].get_weights()
    N = abweights[0].shape[0]
    dim2 = abweights[0].shape[-1]
    idx2a = int(dim2*0.5)
    idx2 = int(dim2*0.7)
    temp = -4*np.ones((N,dim2))
    temp[:,0:idx2a] = 4.
    temp[:,idx2a:idx2] = 0.
    newweight = [temp,]
    m.layers[idx1].set_weights(newweight)
    return m
def initializeab(m):
    idx1 = getlayeridx(m, 'f2a')
    abweights = m.layers[idx1].get_weights()
    dim2 = abweights[0].shape[-1]
    idx2a = int(dim2*0.5)
    idx2 = int(dim2*0.7)
    temp = 1*np.ones((dim2))
    temp[0:idx2a] = 4.
    temp[idx2a:idx2] = 1.
    temp2 = 2*np.ones((dim2))
    newweight = [temp, temp2]
    m.layers[idx1].set_weights(newweight)
    return m
def passweightsm2(m,m2,idxz):
    temp = m.get_weights()
    zs2 = getlayerweights(m2,'f1a')
    temp[idxz] = zs2[0]
    m2.set_weights(temp)
    return m2

def mcpredNauxi(m2,images2,auxi2, dimcls, nsample):
    Nall = images2.shape[0]
    h1 = images2.shape[1]
    w1 = images2.shape[2]
    result = np.zeros((nsample,Nall,h1,w1,dimcls))
    """
    for sample in range(nsample):
        preds = m2.predict([images2,auxiliary2])
        result[sample,:] = np.copy(preds[0])
    meanpred1s = np.mean(result, axis=0)
    meanpred1s = np.exp(meanpred1s)/np.sum(np.exp(meanpred1s), axis=-1, keepdims=True)
    """
    for sample in range(nsample):
        preds = m2.predict([images2,auxi2])
        logit = preds[0]
        logit = logit - np.max(logit, axis=-1, keepdims=True)
        softmax = np.exp(logit)/np.sum(np.exp(logit), axis=-1, keepdims=True)
        result[sample,:] = np.copy(softmax)
    meanpred1s = np.mean(result, axis=0)
    temp1 = meanpred1s * np.log(meanpred1s + 1e-8)
    uncertainty1 = - np.sum(temp1, axis=-1)
    classpreds = meanpred1s.argmax(-1)
    maxentropy = - np.log(1/dimcls+1e-8)
    uncertainty0s = uncertainty1 / maxentropy
    return [meanpred1s, classpreds, uncertainty0s]

def preparePriors3(m, reducevalue, reducevaluelnow, lW, lW2):

    priors = []
    for idx in range(len(lW)-1):        
        idxW = getlayeridx(m, lW[idx])
        wweights = m.layers[idxW].get_weights()
        wpriormu = wweights[0]
        wpriorvar = np.exp(wweights[1]-reducevalue)
        bpriormu = wweights[2]
        bpriorvar = np.exp(wweights[3]-reducevalue)
        priors.append([wpriormu, wpriorvar, bpriormu, bpriorvar])
    idxZ = getlayeridx(m, lW[-1])
    abzweights = m.layers[idxZ].get_weights()
    vpriora = np.log(np.exp(abzweights[0])+1)
    vpriorb = np.log(np.exp(abzweights[1])+1)
    priors.append([vpriora,vpriorb])
        
    priors2 = []
    for idx in range(len(lW2)):        
        idxW = getlayeridx(m, lW2[idx])
        wweights = m.layers[idxW].get_weights()
        wpriormu = wweights[0]
        wpriorvar = np.exp(wweights[1]-reducevalue)
        wpriormup = wweights[2]
        wpriorvarp = np.exp(wweights[3]-reducevalue)
        bpriormu = wweights[4]
        bpriorvar = np.exp(wweights[5]-reducevalue)
        priors2.append([wpriormu, wpriorvar, wpriormup, wpriorvarp, bpriormu, bpriorvar])
    return [priors,priors2]


def initializez(m):
    idx1 = getlayeridx(m, 'f2a')
    abweights = m.layers[idx1].get_weights()
    dim2 = abweights[0].shape[-1]
    idx2a = int(dim2*0.5)
    idx2 = int(dim2*0.7)
    temp = 1*np.ones((dim2))
    temp[0:idx2a] = 4.
    temp[idx2a:idx2] = 1.
    temp2 = 2*np.ones((dim2))
    newweight = [temp, temp2]
    m.layers[idx1].set_weights(newweight)
    return m
def initializedense(m):
    idx1 = getlayeridx(m, 'f1a')
    abweights = m.layers[idx1].get_weights()
    dim2 = abweights[0].shape[-1]
    idx2a = int(dim2*0.5)
    idx2 = int(dim2*0.7)
    temp = -4*np.ones((dim2))
    temp[0:idx2a] = 4.
    temp[idx2a:idx2] = 0.
    newweight = [abweights[0],abweights[1], temp,abweights[3]]
    m.layers[idx1].set_weights(newweight)
    return m


def smartreduceVar12(m, lW, lW2):
    for idx in range(len(lW)):        
        idxW = getlayeridx(m, lW[idx])
        wweights = m.layers[idxW].get_weights()
        newweights = [wweights[0],wweights[1]-4,wweights[2], wweights[3]-4]
        m.layers[idxW].set_weights(newweights)
    for idx in range(len(lW2)):        
        idxW = getlayeridx(m, lW2[idx])
        wweights = m.layers[idxW].get_weights()
        newweights = [wweights[0],wweights[1]-4,wweights[2], wweights[3]-4]
        m.layers[idxW].set_weights(newweights)
    return m

def preparePriors12(m, reducevalue, lW, lW2):

    priors = []
    priors2 = []
    for idx in range(len(lW)):        
        idxW = getlayeridx(m, lW[idx])
        wweights = m.layers[idxW].get_weights()
        wpriormu = wweights[0]
        wpriorvar = np.exp(wweights[1]-reducevalue)
        bpriormu = wweights[2]
        bpriorvar = np.exp(wweights[3]-reducevalue)
        priors.append([wpriormu, wpriorvar, bpriormu, bpriorvar])
    for idx in range(len(lW2)):        
        idxW = getlayeridx(m, lW2[idx])
        wweights = m.layers[idxW].get_weights()
        wpriormu = wweights[0]
        wpriorvar = np.exp(wweights[1]-reducevalue)
        wpriormu2 =  wweights[2]
        wpriorvar2 = np.exp(wweights[3]-reducevalue)
        bpriormu = wweights[4]
        bpriorvar = np.exp(wweights[5]-reducevalue)
        priors2.append([wpriormu, wpriorvar, wpriormu2, wpriorvar2, bpriormu, bpriorvar])
        
    idxW = getlayeridx(m, 'f2a')
    wweights = m.layers[idxW].get_weights()
    vpriora = np.log(np.exp(wweights[0])+1)
    vpriorb = np.log(np.exp(wweights[1])+1)
    priors.append([vpriora, vpriorb])
    return [priors,priors2]

def mcpredNall(m2,images2, dimcls, nsample):
    Nall = images2.shape[0]
    h1 = images2.shape[1]
    w1 = images2.shape[2]
    result = np.zeros((nsample,Nall,h1,w1,dimcls))
    """
    for sample in range(nsample):
        preds = m2.predict([images2,auxiliary2])
        result[sample,:] = np.copy(preds[0])
    meanpred1s = np.mean(result, axis=0)
    meanpred1s = np.exp(meanpred1s)/np.sum(np.exp(meanpred1s), axis=-1, keepdims=True)
    """
    for sample in range(nsample):
        preds = m2.predict([images2,])
        logit = preds[0]
        logit = logit - np.max(logit, axis=-1, keepdims=True)
        softmax = np.exp(logit)/np.sum(np.exp(logit), axis=-1, keepdims=True)
        result[sample,:] = np.copy(softmax)
    meanpred1s = np.mean(result, axis=0)
    temp1 = meanpred1s * np.log(meanpred1s + 1e-8)
    uncertainty1 = - np.sum(temp1, axis=-1)
    classpreds = meanpred1s.argmax(-1)
    maxentropy = - np.log(1/dimcls+1e-8)
    uncertainty0s = uncertainty1 / maxentropy
    return [meanpred1s, classpreds, uncertainty0s]



def initializezAll(m, dims, ndata, layernames):
    
    for i in range(len(layernames)):
        layername = layernames[i]
        idx1 = getlayeridx(m, layername)
        abzweights = m.layers[idx1].get_weights()
        
        dim2 = dims[-1][i]
        idx2a = int(dim2*0.4)
        idx2 = int(dim2*0.5)
        temp = 0.05*np.ones((1,dim2))
        temp[0,0:idx2a] = 0.95
        temp[0,idx2a:idx2] = 0.5
        temp = np.log(temp / (1-temp))
        zweight = np.tile(temp, [ndata,1])
    
        newweight = [abzweights[0], abzweights[1], zweight]
        m.layers[idx1].set_weights(newweight)
    return m

def deBN(bnweights, convweights):
    epsilon = 0.001 
    
    gamma = bnweights[0]
    beta = bnweights[1]
    mamean = bnweights[2]
    mavar = bnweights[3]
    conv = convweights[0]
    bias = convweights[2]
    
    convvar = convweights[1] - 2
    biasvar = convweights[3] - 2

    temp = gamma / np.sqrt(mavar+epsilon)
    conv2 = conv * temp
    bias2 = (bias-mamean) * temp + beta
    temp2 = 2*np.log(temp)
    convvar2 = convvar + temp2
    biasvar2 = biasvar + temp2
    return [conv2, convvar2, bias2, biasvar2]

def modeldeBN_noz(m, m3, lW, layernameZ):
    for layername in lW:
        idx1 = getlayeridx(m, layername)
        idx2 = getlayeridx(m3, layername)
        convweights = m.layers[idx1].get_weights()
        bnweights = m.layers[idx1+1].get_weights()
        newweight = deBN(bnweights, convweights)
        m3.layers[idx2].set_weights(newweight)
    for layername in layernameZ:
        idx1 = getlayeridx(m, layername)
        idx2 = getlayeridx(m3, layername)
        newweight = m.layers[idx1].get_weights()
        m3.layers[idx2].set_weights(newweight)
    
    newweight = getlayerweights(m, 'lnow')
    idx2 = getlayeridx(m3, layername)
    m3.layers[idx2].set_weights(newweight)
    return m3


def exportZ(m1, layernames):
    zs = []
    for layername in layernames:
        idx1 = getlayeridx(m1, layername)
        abzweights = m1.layers[idx1].get_weights()
        
        target = abzweights[2]
        zs.append(target)
    return zs
def exportZm3(m1, layernames):
    zs = []
    for layername in layernames:
        idx1 = getlayeridx(m1, layername+'a')
        abzweights = m1.layers[idx1].get_weights()
        
        target = abzweights[2]
        zs.append(target)
    return zs
