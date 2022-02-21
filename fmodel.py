import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import Reshape, Flatten, RepeatVector, Concatenate, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, DepthwiseConv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda, Multiply, Add
from tensorflow.keras.layers import Softmax, ReLU
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras import regularizers

initialvar = -8.
small = 1e-5


class ConvBayes(layers.Layer):
    def __init__(self, dims, priors, trns, idx, kldiv, wh, name='name'):
        super().__init__(name=name)
        self.dim1 = dims[0][idx]
        self.dim2 = dims[1][idx]
        self.trainable = trns
        self.wmu0 = priors[idx][0]
        self.wvar0 = priors[idx][1]
        self.bmu0 = priors[idx][2]
        self.bvar0 = priors[idx][3]
        self.wh = wh
        self.kldiv = kldiv
    def build(self, inshape):
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[self.wh,self.wh,self.dim1, self.dim2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.wh,self.wh,self.dim1, self.dim2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.dim2])
        self.bv = self.add_weight("bv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.dim2])
    def call(self, x):
        rngw = tf.random.truncated_normal([self.wh,self.wh,self.dim1,self.dim2])
        rngb = tf.random.truncated_normal([self.dim2])
        wnow = self.w + tf.math.multiply(rngw, tf.exp(0.5*self.wv))
        bnow = self.b + tf.math.multiply(rngb, tf.exp(0.5*self.bv))
        tempout = tf.add(tf.nn.conv2d(x, wnow, [1,1,1,1], padding='SAME'), bnow)
        
        term0 = -0.5*self.wh*self.wh*self.dim1*self.dim2
        term1 = 0.5*tf.reduce_sum(np.log(self.wvar0) - self.wv)
        term2 = 0.5*tf.reduce_sum((tf.exp(self.wv) + (self.w - self.wmu0)**2) / self.wvar0)
        term0b = -0.5*self.dim2
        term1b = 0.5*tf.reduce_sum(np.log(self.bvar0) - self.bv)
        term2b = 0.5*tf.reduce_sum((tf.exp(self.bv) + (self.b - self.bmu0)**2) / self.bvar0)
        sumkl = term1 + term2 + term0 + term1b + term2b + term0b
        self.add_loss(sumkl / self.kldiv)
        
        return tempout
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1], inshape[2], self.dim2)


class LBayes(layers.Layer):
    def __init__(self, dims, priors, trns, idx, kldiv, name='name'):
        super().__init__(name=name)
        self.dim2 = dims
        self.trainable = trns
        self.wmu0 = priors[idx][0]
        self.wvar0 = priors[idx][1]
        self.bmu0 = priors[idx][2]
        self.bvar0 = priors[idx][3]
        self.kldiv = kldiv
    def build(self, inshape):
        self.dim1 = inshape[-1]
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[self.dim1, self.dim2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.dim1, self.dim2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.dim2])
        self.bv = self.add_weight("bv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.dim2])
    def call(self, x):
        rngw = tf.random.truncated_normal([self.dim1,self.dim2])
        rngb = tf.random.truncated_normal([self.dim2])
        wnow = self.w + tf.math.multiply(rngw, tf.exp(0.5*self.wv))
        bnow = self.b + tf.math.multiply(rngb, tf.exp(0.5*self.bv))
        tempout = tf.add(tf.matmul(x, wnow), bnow)
        
        term0 = -0.5*self.dim1*self.dim2
        term1 = 0.5*tf.reduce_sum(np.log(self.wvar0) - self.wv)
        term2 = 0.5*tf.reduce_sum((tf.exp(self.wv) + (self.w - self.wmu0)**2) / self.wvar0)
        term0b = -0.5*self.dim2
        term1b = 0.5*tf.reduce_sum(np.log(self.bvar0) - self.bv)
        term2b = 0.5*tf.reduce_sum((tf.exp(self.bv) + (self.b - self.bmu0)**2) / self.bvar0)
        sumkl = term1 + term2 + term0 + term1b + term2b + term0b
        self.add_loss(sumkl / self.kldiv)
        
        return tf.nn.softmax(tempout)
    def compute_output_shape(self, inshape):
        return (inshape[0], self.dim2)



class ConvMobile(layers.Layer):
    def __init__(self, dims, priors2, trns, idx, idx2, kldiv, wh, name='name'):
        super().__init__(name=name)
        self.dim1 = dims[0][idx]
        self.dim2 = dims[1][idx]
        self.trainable = trns
        self.wmu0 = priors2[idx2][0]
        self.wvar0 = priors2[idx2][1]
        self.bmu0 = priors2[idx2][4]
        self.bvar0 = priors2[idx2][5]
        self.wmu0p = priors2[idx2][2]
        self.wvar0p = priors2[idx2][3]
        self.wh = wh
        self.kldiv = kldiv
    def build(self, inshape):
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[self.wh,self.wh,self.dim1, 1])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.wh,self.wh,self.dim1, 1])
        self.wp = self.add_weight("wp", trainable=self.trainable,
                                 initializer=w_init, shape=[1,1,self.dim1, self.dim2])
        self.wvp = self.add_weight("wvp", trainable=self.trainable,
                                 initializer=v_init, shape=[1,1,self.dim1, self.dim2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.dim2])
        self.bv = self.add_weight("bv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.dim2])
    def call(self, x):
        rngw = tf.random.truncated_normal([self.wh,self.wh,self.dim1,1])
        rngwp = tf.random.truncated_normal([1,1,self.dim1,self.dim2])
        rngb = tf.random.truncated_normal([self.dim2])
        wnow = self.w + tf.math.multiply(rngw, tf.exp(0.5*self.wv))
        wpnow = self.wp + tf.math.multiply(rngwp, tf.exp(0.5*self.wvp))
        bnow = self.b + tf.math.multiply(rngb, tf.exp(0.5*self.bv))
        
        outdepth = tf.nn.depthwise_conv2d(x, wnow, [1,1,1,1], padding='SAME')
        outpoint = tf.add(tf.nn.conv2d(outdepth, wpnow, [1,1,1,1], padding='SAME'), bnow)
        
        term0 = -0.5*self.wh*self.wh*self.dim1
        term1 = 0.5*tf.reduce_sum(np.log(self.wvar0) - self.wv)
        term2 = 0.5*tf.reduce_sum((tf.exp(self.wv) + (self.w - self.wmu0)**2) / self.wvar0)
        term0a = -0.5*self.dim1*self.dim2
        term1a = 0.5*tf.reduce_sum(np.log(self.wvar0p) - self.wvp)
        term2a = 0.5*tf.reduce_sum((tf.exp(self.wvp) + (self.wp - self.wmu0p)**2) / self.wvar0p)
        term0b = -0.5*self.dim2
        term1b = 0.5*tf.reduce_sum(np.log(self.bvar0) - self.bv)
        term2b = 0.5*tf.reduce_sum((tf.exp(self.bv) + (self.b - self.bmu0)**2) / self.bvar0)
        sumkl = term1 + term2 + term0 + term1a + term2a + term0a + term1b + term2b + term0b
        self.add_loss(sumkl / self.kldiv)
        
        return outpoint
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1], inshape[2], self.dim2)


class AveMobile(layers.Layer):
    def __init__(self, name='name'):
        super().__init__(name=name)
    def call(self, x):
        ws = tf.expand_dims(tf.expand_dims(tf.expand_dims(x[1],1),1),1)
        tempout = tf.reduce_sum(x[0]*ws, axis=-1)
        return tempout
    def compute_output_shape(self, inshape):
        return (inshape[0][0], inshape[0][1], inshape[0][2], inshape[0][1])

class AllZ(layers.Layer):
    def __init__(self, param, trns, name='layername'):
        super().__init__(name=name)
        self.N = param[0]
        self.nbatch = param[1]
        self.nanc = param[2]
        self.w = param[3] 
        self.trainable = trns
    def build(self, inshape): 
        z_init = tf.keras.initializers.Constant(0.)
        self.zs = self.add_weight("z",trainable=self.trainable,
                                 initializer=z_init, shape=[self.N, self.nanc],)
    def call(self, xs):
        idxstart = tf.cast(xs[0,0], tf.int32)
        z = self.zs[idxstart: idxstart+self.nbatch,:]
        output = tf.math.softmax(z, axis=-1)
        temp = tf.reduce_sum(tf.square(z)) * self.w
        self.add_loss(temp)
        return output
    def compute_output_shape(self, inshape):
        return (self.nbatch,self.nanc)  
class AllZshare(layers.Layer):
    def __init__(self, param, trns, name='layername'):
        super().__init__(name=name)
        self.N = param[0]
        self.nbatch = param[1]
        self.nanc = param[2]
        self.w = param[3] 
        self.trainable = trns
    def build(self, inshape): 
        z_init = tf.keras.initializers.Constant(0.)
        self.zs = self.add_weight("z",trainable=self.trainable,
                                 initializer=z_init, shape=[1, self.nanc],)
    def call(self, xs):
        z = tf.tile(self.zs, [self.nbatch, 1])
        output = tf.math.softmax(z, axis=-1)
        temp = tf.reduce_sum(tf.square(z)) * self.w
        self.add_loss(temp)
        return output
    def compute_output_shape(self, inshape):
        return (self.nbatch,self.nanc) 


class sampleZ(layers.Layer):
    def __init__(self, dims, priors, trns, idx, kldiv,kldiv2, name='layername'):
        super().__init__(name=name)
        self.priora = priors[idx][0]
        self.priorb = priors[idx][1]
        self.kldiv = kldiv
        self.kldiv2 = kldiv2 
        self.trainable = trns
        self.temperature = 1
    def build(self, inshape): 
        self.nbatch = inshape[0]
        self.nanc = inshape[1] 
        a_init = tf.keras.initializers.Constant(5.)
        b_init = tf.keras.initializers.Constant(1.)
        self.a = self.add_weight("a",trainable=self.trainable,
                                 initializer=a_init,
                                  shape=[self.nanc],)
        self.b = self.add_weight("b",trainable=self.trainable,
                                 initializer=b_init,
                                  shape=[self.nanc],)
    def call(self, xs):
        alpha = tf.nn.softplus(self.a)
        beta = tf.nn.softplus(self.b)
        epsab = tf.random.uniform(shape=tf.shape(alpha), minval=0.1, maxval=0.9)
        kuma = tf.math.pow(1 - tf.math.pow(epsab, 1/beta), 1/alpha)
        temp1 = tf.math.log(kuma + small)
        temp21 = tf.math.log(1-kuma + small)
        temp22 = tf.math.cumsum(temp21, axis=-1)
        temp23 = tf.concat([tf.constant([0.,]), temp22[0:self.nanc-1]],axis=0)
        priorlogpie = temp1+temp23
        
        ab = alpha+beta
        term0 = tf.math.lgamma(self.priora)+tf.math.lgamma(self.priorb)-tf.math.lgamma(self.priora+self.priorb)
        term1 = -tf.math.lgamma(alpha)-tf.math.lgamma(beta)+tf.math.lgamma(ab)
        term2 = tf.multiply(alpha-self.priora, tf.math.digamma(alpha))
        term3 = tf.multiply(beta-self.priorb, tf.math.digamma(beta))
        term4 = tf.multiply(-ab+self.priora+self.priorb, tf.math.digamma(ab))
        sumklbeta = tf.reduce_sum(term0+term1+term2+term3+term4)
        self.add_loss( sumklbeta / (self.kldiv) )
        temp = xs
        postlogpie = tf.math.log(temp+small)
        term11 = tf.multiply(temp, postlogpie - priorlogpie)
        sumklcat = tf.reduce_sum(term11)
        self.add_loss( sumklcat / (self.kldiv2) )
        
        epspie = tf.random.uniform(shape=tf.shape(xs), minval=0.1, maxval=0.9)
        logit = (postlogpie + tf.math.log(-tf.math.log(epspie))) / self.temperature
        output = tf.math.softmax(logit)
        
        return output
    def compute_output_shape(self, inshape):
        return inshape


def convblockbnvanilla(inputs, dims, priors, trns, kldiv, mmt, bn, idx, prefix):
    b1a = ConvBayes(dims, priors, trns, idx, kldiv,3,name=prefix+'a')(inputs)
    b1ad = BatchNormalization(momentum=mmt, trainable=bn, name=prefix+'ad')(b1a, training=bn) 
    b1ar = ReLU(name=prefix+'ar')(b1ad)
    b1b = ConvBayes(dims, priors, trns, idx+1, kldiv,3,name=prefix+'b')(b1ar)
    b1br = ReLU(name=prefix+'br')(b1b)
    return b1br


def myNNbnZ_vanilla(datas, dims, npat, priors, priors2, trns, kldivs, mmt, bn):
    l2lambda = 1e-8
    reg = regularizers.l2(l2=l2lambda)
    kldiv = kldivs[0]
    kldiv2 = kldivs[1]
    trn1 = trns[0]
    trn2 = trns[1]
    
    b1 = convblockbnvanilla(datas[0], dims, priors, trn1, kldiv, mmt, bn, 0, 'b1')
    p1 = MaxPooling2D((2, 2), strides=(2, 2), name='p1')(b1)
    
    b2 = convblockbnvanilla(p1, dims, priors, trn1, kldiv, mmt, bn, 2, 'b2')
    p2 = MaxPooling2D((2, 2), strides=(2, 2), name='p2')(b2)
    
    b3 = convblockbnvanilla(p2, dims, priors, trn1, kldiv, mmt, bn, 4, 'b3')
    p3 = MaxPooling2D((2, 2), strides=(2, 2), name='p3')(b3)
    
    b4 = convblockbnvanilla(p3, dims, priors, trn1, kldiv, mmt, bn, 6, 'b4')
    p4 = UpSampling2D(size=(2, 2), name='p4')(b4)
    
    b5e = Concatenate(name='b5e')([p4, b3])
    b5 = convblockbnvanilla(b5e, dims, priors, trn1, kldiv, mmt, bn, 8, 'b5')
    p5 = UpSampling2D(size=(2, 2), name='p5')(b5)
    
    b6e = Concatenate(name='b6e')([p5, b2])
    b6 = convblockbnvanilla(b6e, dims, priors, trn1, kldiv, mmt, bn, 10, 'b6')
    p6 = UpSampling2D(size=(2, 2), name='p6')(b6)
    
    b7e = Concatenate(name='b7e')([p6, b1])
    b7 = convblockbnvanilla(b7e, dims, priors, trn1, kldiv, mmt, bn, 12, 'b7')
    
    lnow = ConvBayes(dims, priors, trn1, 14, kldiv, 1, name='lnow')(b7)
    lmean = Lambda(lambda x: x, name='lmean') (lnow)
    
    return [lnow, lmean]


def convblock1(inputs, dims, priors, trns, kldiv, idx, prefix):
    b1a = ConvBayes(dims, priors, trns, idx, kldiv,3,name=prefix+'a')(inputs)
    b1ar = ReLU(name=prefix+'ar')(b1a)
    b1b = ConvBayes(dims, priors, trns, idx+1, kldiv,3,name=prefix+'b')(b1ar)
    b1br = ReLU(name=prefix+'br')(b1b)
    return b1br
def convblock2(inputs, dims, priors, priors2, npat, trns, kldiv, idx,idx2, prefix):
    trn1 = trns[0]
    trn2 = trns[1]
    trn10 = trns[2] 
    b1a = ConvBayes(dims, priors, trn1, idx, kldiv,3,name=prefix+'a')(inputs[0])
    b1ar = ReLU(name=prefix+'ar')(b1a)
    b1b = ConvBayes(dims, priors, trn1, idx+1, kldiv,3,name=prefix+'b')(b1ar)
    b1br = ReLU(name=prefix+'br')(b1b)
    
    llist = []
    for i in range(npat):
        l = ConvMobile(dims, priors2, trn10[i], idx+1,idx2+i, kldiv, 3, name=prefix+'m'+str(i))(b1ar)
        llist.append(l)
    stack = Lambda(lambda x: tf.stack(x, axis=-1), name=prefix+'s')(llist)
    avemobile = AveMobile(name=prefix+'ma')([stack, inputs[1]])
    b1c = Lambda(lambda x: x[0]+x[1], name=prefix+'c')([b1br, avemobile])
    return b1c

def adaZpretrain(datas, dims, params, priors, priors2, trns, kldivs):
    npat = params[2]
    kldiv = kldivs[0]
    kldiv2 = kldivs[1]
    trn1 = trns[0]
    trn2 = trns[1]
    trnaux = trns[2]
    trnz = trns[3]
    trnab = trns[4]
    
    b1 = convblock1(datas[0], dims, priors, trn1, kldiv, 0, 'b1')
    p1 = MaxPooling2D((2, 2), strides=(2, 2), name='p1')(b1)
    
    b2 = convblock1(p1, dims, priors, trn1, kldiv, 2, 'b2')
    p2 = MaxPooling2D((2, 2), strides=(2, 2), name='p2')(b2)
    
    b3 = convblock1(p2, dims, priors, trn1, kldiv, 4, 'b3')
    p3 = MaxPooling2D((2, 2), strides=(2, 2), name='p3')(b3)
    
    b4 = convblock1(p3, dims, priors, trn1, kldiv, 6, 'b4')
    p4 = UpSampling2D(size=(2, 2), name='p4')(b4)
    
    f1a = AllZ(params, trnz, name='f1a')(datas[1])
    f2a = sampleZ(dims, priors, trnab, 15, kldiv,kldiv2, name='f2a')(f1a)
    
    b5e = Concatenate(name='b5e')([p4, b3])
    b5 = convblock2([b5e,f2a], dims, priors, priors2, npat, trns, kldiv, 8,0, 'b5')
    p5 = UpSampling2D(size=(2, 2), name='p5')(b5)
    
    b6e = Concatenate(name='b6e')([p5, b2])
    b6 = convblock2([b6e,f2a], dims, priors, priors2, npat, trns, kldiv, 10,npat*1, 'b6')
    p6 = UpSampling2D(size=(2, 2), name='p6')(b6)
    
    b7e = Concatenate(name='b7e')([p6, b1])
    b7 = convblock2([b7e,f2a], dims, priors, priors2, npat, trns, kldiv, 12,npat*2, 'b7')
    
    lnow = ConvBayes(dims, priors, trn1, 14, kldiv, 1, name='lnow')(b7)
    lmean = Lambda(lambda x: x, name='lmean') (lnow)
    
    return [lnow, lmean]



def adaZshare(datas, dims, params, priors, priors2, trns, kldivs):
    npat = params[2]
    kldiv = kldivs[0]
    kldiv2 = kldivs[1]
    trn1 = trns[0]
    trn2 = trns[1]
    trnaux = trns[2]
    trnz = trns[3]
    trnab = trns[4]
    
    b1 = convblock1(datas[0], dims, priors, trn1, kldiv, 0, 'b1')
    p1 = MaxPooling2D((2, 2), strides=(2, 2), name='p1')(b1)
    
    b2 = convblock1(p1, dims, priors, trn1, kldiv, 2, 'b2')
    p2 = MaxPooling2D((2, 2), strides=(2, 2), name='p2')(b2)
    
    b3 = convblock1(p2, dims, priors, trn1, kldiv, 4, 'b3')
    p3 = MaxPooling2D((2, 2), strides=(2, 2), name='p3')(b3)
    
    b4 = convblock1(p3, dims, priors, trn1, kldiv, 6, 'b4')
    p4 = UpSampling2D(size=(2, 2), name='p4')(b4)
    
    f1a = AllZshare(params, trnz, name='f1a')(datas[1])
    f2a = sampleZ(dims, priors, trnab, 15, kldiv,kldiv2, name='f2a')(f1a)
    
    b5e = Concatenate(name='b5e')([p4, b3])
    b5 = convblock2([b5e,f2a], dims, priors, priors2, npat, trns, kldiv, 8,0, 'b5')
    p5 = UpSampling2D(size=(2, 2), name='p5')(b5)
    
    b6e = Concatenate(name='b6e')([p5, b2])
    b6 = convblock2([b6e,f2a], dims, priors, priors2, npat, trns, kldiv, 10,npat*1, 'b6')
    p6 = UpSampling2D(size=(2, 2), name='p6')(b6)
    
    b7e = Concatenate(name='b7e')([p6, b1])
    b7 = convblock2([b7e,f2a], dims, priors, priors2, npat, trns, kldiv, 12,npat*2, 'b7')
    
    lnow = ConvBayes(dims, priors, trn1, 14, kldiv, 1, name='lnow')(b7)
    lmean = Lambda(lambda x: x, name='lmean') (lnow)
    
    return [lnow, lmean]
