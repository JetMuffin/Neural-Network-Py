import numpy as np
import sys
import math

training_data = np.array([])
labels_data = np.array([])
w = np.array([])
b = 0
hardlim = lambda x: 1 if x >= 0 else 0

# scale parameter p to range(0,1)
def scale(data):
    shape = data.shape
    temp = np.zeros(shape)
    for i in range(data.ndim):
        row_square_sum = sum([j*j for j in data[i]])
        temp[i] = data[i]/math.sqrt(row_square_sum)
    return temp

# calculate parameter w using formula W=TP^T
def cal():
    scaled_training_data = scale(training_data)
    w = np.dot(labels_data, scaled_training_data)
    return w

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: python Hebb.py train_data labels_data modelFile"
        exit(0)

    # read data
    training_data = np.loadtxt(sys.argv[1], delimiter=" ")
    labels_data = np.loadtxt(sys.argv[2], delimiter=" ")

    # calculate
    w = cal()
    print "Result W:",w
    np.savetxt(sys.argv[3], w)