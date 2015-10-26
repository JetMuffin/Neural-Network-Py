# -*- coding: utf-8 -*-
import math  
import random 
import numpy as np

random.seed(0)  
logsig = lambda x:1/(1+math.exp(-x))

# 计算[a,b]间的随机值
def rand(a, b):  
    return (b-a)*random.random() + a  

def purelin(data):
    return data

class BPNN:
    def __init__(self, ni, nh, no):
        self.ni = ni 
        self.nh = nh  
        self.no = no  

        self.ai = [1.0]*self.ni  
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no  

        self.wi = np.zeros((self.ni, self.nh), np.float32)
        self.wo = np.zeros((self.nh, self.no), np.float32)
        self.bi = np.zeros((self.ni, ), np.float32)
        self.bo = np.zeros((self.no, ), np.float32)

        for i in range(self.ni):  
            for j in range(self.nh):  
                self.wi[i][j] = rand(-0.2, 0.2)  
        for i in range(self.nh):  
            for j in range(self.no):  
                self.wo[i][j] = rand(-0.2, 0.2) 
        for i in range(self.ni):  
                self.bi[i] = rand(-0.2, 0.2) 
        for i in range(self.no):  
                self.bo[i] = rand(-0.2, 0.2)

    def forward(self, p):
        self.ai = p

        # 计算隐层
        # self.ah = np.dot(self.ai, self.wi)
        self.ah = np.dot(self.ai, self.wi)+self.bi
        for i in range(self.ah.shape[0]):
            self.ah[i] = logsig(self.ah[i])

        # 计算输出层
        # self.ao = np.dot(self.ah, self.wo)
        self.ao = np.dot(self.ah, self.wo)+self.bo
        return self.ao

    def back(self, target, alpha):
        # 计算反向敏感性
        so = -2*(target - self.ao)
        mat = np.eye(self.ah.shape[0])
        for i in range(self.ah.shape[0]):
            mat[i][i] = (1-self.ah[i])*self.ah[i]
        mat = np.dot(mat, self.wo)
        si = np.dot(mat, so)

        # 更新权值
        wo = np.mat(self.wo) - alpha*so[0]*np.mat(self.ah).T
        self.wo = np.array(wo)
        self.wi = np.array(self.wi - np.mat(self.ai).T*np.mat(si))
        # self.bo = self.bo - alpha*so
        # self.bi = self.bi - alpha*si

        # 返回误差贡献
        return (target-self.ao)**2

    def train(self, p, t, iterations=5000, alpha=0.8):  
        iter_count = 0
        for i in range(iterations):  
            error = 0.0  
            iter_count += 1
            length = p.shape[0]
            for j in xrange(length):
                self.forward(p[j])  
                error = error + self.back(t[j], alpha)
            if error < 0.00001:
                break
            if i % 100 == 0:
                print "error: ", error
        print "iter_count: ",iter_count

    def test(self, p):
        res = self.forward(p)
        print p,"->",res

def demo():
    p = np.array([[0,0],[0,1],[1,0],[1,1]])
    t = np.array([[0],[1],[1],[0]])
    a = np.array([0,1])
    b = np.array([1,1])
    c = np.array([1,0])
    d = np.array([0,0])
    model = BPNN(2,2,1)
    model.train(p, t)
    model.test(a)
    model.test(b)
    model.test(c)
    model.test(d)

if __name__ == '__main__':
    demo()