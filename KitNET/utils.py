
import numpy
from scipy.stats import norm
numpy.seterr(all='ignore')

def pdf(x,mu,sigma): # 正态分布的概率密度函数（pdf）
    x = (x-mu)/sigma
    return numpy.exp(-x**2/2)/(numpy.sqrt(2*numpy.pi)*sigma)

def invLogCDF(x,mu,sigma): # 正态分布的累积分布函数（cdf）
    x = (x - mu) / sigma
    return norm.logcdf(-x) # 注意：归一化后乘以 -1，是为了直接拿到 1-cdf（越小越异常）

def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


def dsigmoid(x):
    return x * (1. - x)

def tanh(x):
    return numpy.tanh(x)

def dtanh(x):
    return 1. - x * x

def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # 减去最大值，防止指数溢出
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # 二维的情形（ndim = 2）


def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

class rollmean:
    def __init__(self,k):
        self.winsize = k
        self.window = numpy.zeros(self.winsize)
        self.pointer = 0

    def apply(self,newval):
        self.window[self.pointer]=newval
        self.pointer = (self.pointer+1) % self.winsize
        return numpy.mean(self.window)

# 高斯分布的概率密度函数（下面这段是已废弃的旧实现，保留仅供参考）
# def gaussian(x, mean=0.0, scale=1.0):
#     s = 2 * numpy.power(scale, 2)
#     e = numpy.exp( - numpy.power((x - mean), 2) / s )

#     return e / numpy.square(numpy.pi * s)

