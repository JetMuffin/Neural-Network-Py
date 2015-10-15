# -*- coding: utf-8 -*-
import numpy as np
import sys
import math
import os
import cv2

training_data = np.array([])
noise_testing_data = np.array([])
hardlims = lambda x: 1 if x >= 0 else -1

# 使用Hebb规则进行计算权值W
def hebb(data):
    w = np.zeros([data.shape[1], data.shape[1]])
    for row in data:
        w += np.outer(row, row.transpose())
    return w

# 求矩阵的逆
def inv(data):
    temp = np.mat(data).T
    temp = (temp.T * temp).I * temp.T
    return temp

# 使用仿逆规则进行计算权值W
def moore_penrose(data):
    p = np.mat(data)
    w = p.T * inv(p)
    w = np.array(w)
    return w

# 开始联想
def associate(w, p):
    a = np.dot(w, p)
    for i in range(a.shape[0]):
        a[i] = hardlims(a[i])
    return a

# 将特征向量转化为50x60的子图
def digit_mask(data):
    mask = np.zeros((60,50),np.uint8)
    for i in range(data.shape[0]):
        if(data[i] == -1):
            mask[i/5*10+1:i/5*10+9, i%5*10+1:i%5*10+9] = 0
        else:
            mask[i/5*10+1:i/5*10+9, i%5*10+1:i%5*10+9] = 255
    return mask  

# 隐藏部分特征
def hide_part_of_digit(data, part):
    data[part*30:] = 1
    return data

# 测试板
def show_on_board(data,w):
    image = np.zeros((300, 600),np.uint8)
    image[:] = 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 显示原始数字特征
    cv2.putText(image,'origin',(8,30), font, 0.7,(0,0,0),2)
    for i in range(data.shape[0]):
        image[i*70+40:i*70+100, 10:60] = digit_mask(data[i])
        
    # 隐藏50%的特征再进行联想
    cv2.putText(image,'hiding 50%',(100,30), font, 0.8,(0,0,0),2)
    cv2.putText(image,'before',(100,255), font, 0.5,(0,0,0),2)
    cv2.putText(image,'after',(185,255), font, 0.5,(0,0,0),2)
    for i in range(data.shape[0]):
        hided_data = hide_part_of_digit(data[i],0.5)
        associted_data = associate(w, hided_data)
        image[i*70+40:i*70+100, 100:150] = digit_mask(hided_data)
        image[i*70+40:i*70+100, 180:230] = digit_mask(associted_data)

    # 隐藏70%的特征再进行联想
    cv2.putText(image,'hiding 70%',(260,30), font, 0.8,(0,0,0),2)
    cv2.putText(image,'before',(260,255), font, 0.5,(0,0,0),2)
    cv2.putText(image,'after',(345,255), font, 0.5,(0,0,0),2)
    for i in range(data.shape[0]):
        hided_data = hide_part_of_digit(data[i],0.30)
        associted_data = associate(w, hided_data)
        image[i*70+40:i*70+100, 260:310] = digit_mask(hided_data)
        image[i*70+40:i*70+100, 340:390] = digit_mask(associted_data)

    # 对带噪声的特征进行联想
    cv2.putText(image,'noise',(450,30), font, 0.8,(0,0,0),2)
    cv2.putText(image,'before',(420,255), font, 0.5,(0,0,0),2)
    cv2.putText(image,'after',(505,255), font, 0.5,(0,0,0),2)    
    for i in range(noise_testing_data.shape[0]):
        image[i*70+40:i*70+100, 420:470] = digit_mask(noise_testing_data[i])
        associted_data = associate(w, noise_testing_data[i])
        image[i*70+40:i*70+100, 500:550] = digit_mask(associted_data)

    cv2.imshow("Image", image)
    cv2.waitKey (0)

if __name__ == '__main__':
    training_data = np.loadtxt('feature_data/self_data')
    noise_testing_data = np.loadtxt('feature_data/noise_self_data')
    w = moore_penrose(training_data)
    show_on_board(training_data,w)
