# -*- coding: utf-8 -*-
import numpy as np
import sys
import math
import os
import cv2

digit_data = np.array([])
char_data = np.array([])
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
def moore_penrose(t, p):
    w = p.T * inv(t)
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
        if(data[i] == 1):
            mask[i/5*10+1:i/5*10+9, i%5*10+1:i%5*10+9] = 0
        else:
            mask[i/5*10+1:i/5*10+9, i%5*10+1:i%5*10+9] = 255
    return mask  

# 隐藏部分特征
def hide_part_of_digit(data, part):
    data[part*30:] = -1
    return data

# 测试板
def show_on_board(digit_data, char_data, w):
    image = np.zeros((700, 650),np.uint8)
    image[:] = 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 显示原始数字特征
    cv2.putText(image,'origin digits',(8,30), font, 0.7,(0,0,0),2)
    for i in range(digit_data.shape[0]):
        image[40:100, 60*i+10:60*i+60] = digit_mask(digit_data[i])
        
    # 显示原始字母特征
    cv2.putText(image,'origin chars',(8,120), font, 0.7,(0,0,0),2)
    for i in range(digit_data.shape[0]):
        image[130:190, 60*i+10:60*i+60] = digit_mask(char_data[i])

    # 不隐藏进行联想
    cv2.putText(image,'hiding 0%',(8,210), font, 0.7,(0,0,0),2)
    for i in range(digit_data.shape[0]):
        image[220:280, 60*i+10:60*i+60] = digit_mask(digit_data[i])
        associted_data = associate(w, digit_data[i])
        image[290:350, 60*i+10:60*i+60] = digit_mask(associted_data)

    # 隐藏15%的特征再进行联想
    cv2.putText(image,'hiding 15%',(8,370), font, 0.7,(0,0,0),2)
    for i in range(digit_data.shape[0]):
        hided_data = hide_part_of_digit(digit_data[i],0.85)
        image[380:440, 60*i+10:60*i+60] = digit_mask(hided_data)
        associted_data = associate(w, hided_data)
        image[450:510, 60*i+10:60*i+60] = digit_mask(associted_data)

    # 隐藏50%的特征再进行联想
    cv2.putText(image,'hiding 50%',(8,530), font, 0.7,(0,0,0),2)
    for i in range(digit_data.shape[0]):
        hided_data = hide_part_of_digit(digit_data[i],0.5)
        image[540:600, 60*i+10:60*i+60] = digit_mask(hided_data)
        associted_data = associate(w, hided_data)
        image[610:670, 60*i+10:60*i+60] = digit_mask(associted_data)

    cv2.imshow("Image", image)
    cv2.waitKey (0)

if __name__ == '__main__':
    digit_data = np.loadtxt('feature_data/non_self_digit_data')
    char_data = np.loadtxt('feature_data/non_self_char_data')
    w = moore_penrose(digit_data, char_data)
    show_on_board(digit_data, char_data, w)
