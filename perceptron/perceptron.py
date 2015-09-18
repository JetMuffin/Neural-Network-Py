import numpy as np
import sys

# parameter init
training_set = []
training_data = np.array([])
labels = []
labels_data = np.array([])
w = np.array([])
b = 0.5
hardlim = lambda x: 1 if x >= 0 else 0

# update parameters 
def update(item, e):
    global w, b
    w += e * item
    b += e

# check whether the classification is correct or not
def check(item, label):
    global w, b
    p = item.transpose()
    a = hardlim(np.dot(w, p) + b)
    e = label - a
    return e

# calculate the parameters to the model
def cal():
    ndim = training_data.ndim
    iter_count = 0    
    while True:
        flag = True
        for i in range(ndim):
            e = check(training_data[i], labels_data[i])
            if(e != 0):
                update(training_data[i], e)
                iter_count += 1
                flag = False
        if(flag):
            break 
    return iter_count   

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: python preceptron.py train_data label_data modelFile"
        exit(0)

    training_data = np.loadtxt(sys.argv[1], delimiter=" ")
    labels_data = np.loadtxt(sys.argv[2], delimiter=" ")
    modelFile = file(sys.argv[3], 'w')
    #set w to zero matrix
    w = np.zeros([1, training_data.shape[1]])

    # train start
    iter_count = cal()

    modelFile.write(" ".join(str(i) for i in w) + "\n")
    modelFile.write(str(b))
    modelFile.close()

    print "After %d iterations, the parameters converge to:"%iter_count
    print "W: ", w
    print "b: ", b    