import numpy as np
import sys
import math

training_data = np.array([])
labels_data = np.array([])
w = np.array([])
b = 0
hardlim = lambda x: 1 if x >= 0 else 0

# calculate inv of matrix using formula p^+ = (p^T*p)^-1*p^T
def inv(data):
    temp = np.zeros(data.shape)
    temp = np.dot(data, data.transpose())
    temp = np.dot(np.linalg.inv(temp), data)
    return temp

# calculate parameter w using formula W=TP^+
def cal():
    inv_training_data = inv(training_data)
    w = np.dot(labels_data, inv_training_data)
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
