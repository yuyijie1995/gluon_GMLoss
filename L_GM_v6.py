import warnings
warnings.filterwarnings('ignore')
import numpy as np
import mxnet.ndarray as nd
import mxnet as mx
from mxnet import gluon,autograd

class L_GM_Loss(gluon.nn.HybridBlock):
    def __init__(self,num_class,feature_dim,alpha,lamda,lr_mult,**kwargs):
        super(L_GM_Loss,self).__init__(**kwargs)
        self._num_class=num_class
        self._feature_dim=feature_dim
        self._alpha=alpha
        self._lamda=lamda
        self.mean=self.params.get('mean',shape=(num_class,feature_dim),init=mx.init.MSRAPrelu(),lr_mult=2)
        self.var = self.params.get('var', shape=(num_class, feature_dim), init=mx.init.Constant(1), lr_mult=lr_mult)
        self.step=0

    def _classification_probability(self,F,x,y,mean,var):
        if self.step==5000:
            self._alpha=self._alpha*2
        if self._alpha>0.3:
            self._alpha=0.3
        N=x.shape[0]
        XY=F.dot(x,mean,transpose_b=True)#(batchsize,classes)
        XX=F.sum(F.square(x),axis=1,keepdims=1)#(bs,1)
        YY=F.sum(F.square(F.transpose(mean)),axis=0,keepdims=1)#(1,classes)
        neg_sqr_dist=-0.5*(XX-2.0*XY+YY)#(bs,classes)
        reshape_neg_sqr_dist=F.reshape(neg_sqr_dist,(-1,self._num_class,1))
        reshape_var=F.reshape(var,(1,self._num_class,-1))
        dist_var=reshape_neg_sqr_dist/(reshape_var+1e-8)#(bs,class,feature_dim)
        margin_dist=F.sum(dist_var,axis=2)
        ALPHA=F.one_hot(y,self._num_class,on_value=self._alpha,dtype='float32')
        K=ALPHA+F.ones((N,self._num_class),dtype='float32')
        logits_with_margin=margin_dist*K
        batch_mean=F.take(mean,y)
        likelihood_reg_loss=self._lamda*(F.sum(F.square(x-batch_mean))/2.0)*(1./N)
        return logits_with_margin,likelihood_reg_loss

    def hybrid_forward(self, F, x, y,mean,var):
        self.step+=1
        if y is None:
            XY = F.dot(x, mean, transpose_b=True)  # (batchsize,classes)
            XX = F.sum(F.square(x), axis=1, keepdims=1)  # (bs,1)
            YY = F.sum(F.square(F.transpose(mean)), axis=0, keepdims=1)  # (1,classes)
            neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)  # (bs,classes)
            reshape_neg_sqr_dist = F.reshape(neg_sqr_dist, (-1, self._num_class, 1))
            reshape_var = F.reshape(var, (1, self._num_class, -1))
            dist_var = reshape_neg_sqr_dist / (reshape_var + 1e-8)  # (bs,class,feature_dim)
            margin_dist = F.sum(dist_var, axis=2)#最后要得到（bs,classes）的logits
            return margin_dist
        logits,likelihood_reg_loss=self._classification_probability(F,x,y,mean,var)
        return logits,likelihood_reg_loss





