# -*- coding: utf-8 -*-
import math  
import random 
import numpy as np
import pylab as pl

random.seed(0)  

# 计算[a,b]间的随机值
def rand(x, y, a, b):  
    mat = np.zeros((x, y), np.float32)
    for i in xrange(mat.shape[0]):
        for j in xrange(mat.shape[1]):
            mat[i][j] = (b-a)*random.random() + a 
    return mat

def logsig(data):
    mat = np.zeros(data.shape, np.float32)
    for i in xrange(data.shape[0]):
        for j in xrange(data.shape[1]):  
            mat[i][j] = 1/(1+math.exp(-data[i,j]))
    return mat

def purelin(data):
    return data

def deri(data):
    mat = np.eye(data.shape[0],data.shape[0])
    for i in range(mat.shape[0]):
        mat[i,i] = (1-data[i,0])*data[i,0]
    return mat

class BPNN:
    def __init__(self, randtype=False):

        #生成权重矩阵
        if randtype:
            self.w1 = rand(2, 2, -0.5, 0.5)
            self.b1 = rand(2, 2, -0.5, 0.5)
            self.w2 = rand(2, 2, -0.5, 0.5)
            self.b2 = rand(2, 2, -0.5, 0.5)
        else:
            self.w1 = np.mat('-0.27; -0.41')
            self.b1 = np.mat('-0.48; -0.13')
            self.w2 = np.mat('0.09 -0.17')
            self.b2 = np.mat('0.48')

    def forward(self, p):
        self.a0 = p
        self.a1 = logsig(self.w1*self.a0 + self.b1)
        self.a2 = purelin(self.w2*self.a1 + self.b2)
        return self.a2

    def back(self, target, alpha):
        # 计算反向传播敏感性
        self.s2 = -2*(target-self.a2)
        self.s1 = deri(self.a1)*self.w2.T*self.s2

        # 根据敏感性调权
        self.w2 = self.w2 - alpha*self.s2*self.a1.T
        self.b2 = self.b2 - alpha*self.s2
        self.w1 = self.w1 - alpha*self.s1*self.a0.T
        self.b1 = self.b1 - alpha*self.s1
        
        return (target-self.a2)**2

    def train(self, p, t, iteration=3000, alpha=0.5):
        length = p.shape[0]
        for times in xrange(iteration):
            error = 0.0  
            for i in xrange(length):
                # 正向计算
                self.forward(p[i, :])
                error = error + self.back(t[i,:], alpha)
            if times%100 == 0:
                print "error: ",error

    def weights(self):
        print "w1: ", self.w1
        print "b1: ", self.b1
        print "w2: ", self.w2
        print "b2: ", self.b2

    def test(self, p):
        res = np.zeros(p.shape, np.float32)
        for i in xrange(p.shape[0]):
            res[i] = self.forward(p[i])
        return res

def demo():
    a = []
    for i in range(50):
        a.append(0.1*i)
    b = [1+math.sin(a[i]*math.pi/4) for i in range(len(a))]
    c = [a[i]+0.1 for i in range(len(a))]
    p = np.mat(a).T
    t = np.mat(b).T
    test = np.mat(c).T
    # p = np.mat('1;2;3;4;5;6;7;8')
    # t = np.mat('1.707;2;1.707;1;0.292;0;0.292;1')
    # test = np.mat('1.3;2.6;3.2;4.7;5.3;6.8;7.9;8.4')
    model = BPNN()
    model.train(p, t)
    res = model.test(test)
    pl.plot(p, t, 'r')
    pl.plot(test, res, 'g')
    pl.show()

if __name__ == '__main__':
    demo()