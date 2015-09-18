import numpy as np
import sys

training_set = []
training_data = np.array([])
labels = []
w = np.array([0.5, -1, -0.5])
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
            e = check(training_data[i], labels[i])
            if(e != 0):
                update(training_data[i], e)
                iter_count += 1
                flag = False
        if(flag):
            break 
    return iter_count   

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: python preceptron.py trainFile modelFile"
        exit(0)

    trainFile = file(sys.argv[1])
    modelFile = file(sys.argv[2], 'w')

    for line in trainFile:
        chunk = line.strip().split(':')
        labels.append(int(chunk[0]))
        data_single = chunk[1].strip().split(' ')
        lens = len(data_single)
        tmp = [int(data_single[i]) for i in range(lens)]
        training_set.append(tmp)
    trainFile.close()

    training_data = np.array(training_set)
    iter_count = cal()

    modelFile.write(" ".join(str(i) for i in w) + "\n")
    modelFile.write(str(b))
    modelFile.close()

    print "After %d iterations, the parameters converge to:"%iter_count
    print "W: ", w
    print "b: ", b    