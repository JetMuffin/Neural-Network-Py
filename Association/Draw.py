# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
import string

img = np.zeros((600, 500), np.uint8)
drawing = False 
index = 0
data = np.array([1 for i in range(30)])
allowed_chars= map( ord,  string.digits+string.letters)

def draw_grid(image):
    image_temp = np.zeros(image.shape)
    for i in range(image_temp.shape[0]): 
        image_temp[i/5*100:i/5*100+99, i%5*100:i%5*100+99] = 1
    return image_temp

def draw_point(event, x, y, flags, params):
    global img, data
    if(event == cv2.EVENT_LBUTTONDOWN):
        index = y/100 * 5 + x/100
        if data[index] == 1:
            img[index/5*100:index/5*100+99, index%5*100:index%5*100+99] = 0
            data[index] = -1
        else:
            img[index/5*100:index/5*100+99, index%5*100:index%5*100+99] = 1
            data[index] = 1

def show_and_ask_for_key():
    cv2.imshow('ground',img)
    key = 128
    key = cv2.waitKey(1)
    while key > 256:
        key %= 256
    return key

def save_labels(result_folder, label):
    output_file = file(result_folder + '/' + label, 'w')
    for i in range(6):
        line = [data[i*5 + j] for j in range(5)]
        output_file.write(" ".join(str(i) for i in line) + '\n')
    output_file.close()

def ground(result_folder):
    done = False
    label = "unknown"
    while True:
        key = show_and_ask_for_key()
        if key == 27: #ESC
            break
        elif key == 10: #Enter
            if done:
                save_labels(result_folder, label)
                break
        elif key in allowed_chars: #Char
            label = unichr(key)
            print "You labeled the image by character ", label
            done = True    

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: python Draw.py result_folder"
        exit(0)

    img = draw_grid(img)

    cv2.namedWindow('ground')
    cv2.setMouseCallback('ground', draw_point)

    label = ground(sys.argv[1])

    cv2.destroyAllWindows()


