# -*- coding: utf-8 -*-
import numpy as np
import sys
import math
import os
import cv2

training_data = np.array([])
feature_size = 30
hardlims = lambda x: 1 if x >= 0 else -1

def hebb(data):
    w = np.zeros([data.shape[1], data.shape[1]])
    for row in data:
        w += np.outer(row, row.transpose())
    return w

def inv(data):
    temp = np.mat(data).T
    temp = (temp.T * temp).I * temp.T
    return temp

def moore_penrose(data):
    p = np.mat(data)
    w = p.T * inv(p)
    w = np.array(w)
    return w

def load(path):
    feature_files = os.listdir(path)
    feature_files.sort()
    feature_length = len(feature_files)
    flag = True
    for feature_file in feature_files:
        temp = np.loadtxt(path + '/' + feature_file).ravel()
        if(flag):
            data = temp
            flag = False
        else:
            data = np.vstack((data, temp))
    return data

def cal(data):
    a = np.dot(w, p)
    for i in range(a.shape[0]):
        a[i] = hardlims(a[i])
    return a

def show(data):
    image = np.zeros([600,500])
    for i in range(data.shape[0]):
        if(data[i] == -1):
            image[i/5*100:i/5*100+99, i%5*100:i%5*100+99] = 0
        else:
            image[i/5*100:i/5*100+99, i%5*100:i%5*100+99] = 1
    cv2.imshow("Image", image)
    cv2.waitKey (0)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: python SelfAssociative.py feature_folder test_data"
        exit(0)

    training_data = load(sys.argv[1])
    print training_data
    w = moore_penrose(training_data)
    p = np.loadtxt(sys.argv[2]).ravel()
    show(p)
    a = cal(p)
    show(a)

