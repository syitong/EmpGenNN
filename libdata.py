import numpy as np

def unit_circle_ideal(gap,label_prob,samplesize):
    X = list()
    Y = list()
    rad1upper = 1 - gap/2
    rad2lower = 1 + gap/2
    for idx in range(samplesize):
        p = np.random.random()
        if p < 0.5:
            theta = np.random.random()*2*np.pi
            radius = np.random.uniform(0,rad1upper)
            X.append(np.array([radius*np.cos(theta),radius*np.sin(theta)]))
            if p < 0.5*label_prob:
                Y.append(-1)
            else:
                Y.append(1)
        if p > 0.5:
            theta = np.random.random()*2*np.pi
            radius = np.random.uniform(rad1upper,2)
            X.append(np.array([radius*np.cos(theta),radius*np.sin(theta)]))
            if p < 0.5 + 0.5*label_prob:
                Y.append(1)
            else:
                Y.append(-1)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def read_MNIST_data(filepath,obs=1000):
    """
    This method is sufficiently general to read in any
    data set of the same structure with MNIST
    """
    with open(filepath,'rb') as f:
        data = f.read()
        offset = 0
        if data[2] < 10:
            length = 1
        elif data[2] < 12:
            length = 2
        elif data[2] < 14:
            length = 4
        elif data[2] < 15:
            length = 8
        dim = list()
        offset = offset + 4
        for idx in range(data[3]):
            dim.append(int.from_bytes(data[3+idx*4+1:3+idx*4+5],'big'))
            offset = offset + 4
        l = length
        for idx in range(1,len(dim)):
            l = l * dim[idx]
        if obs > 0 and obs < dim[0]:
            dim[0] = obs
        X = np.empty((dim[0],l))
        for idx in range(dim[0]):
            for jdx in range(l):
                index = offset + idx * l + jdx
                X[idx,jdx] = data[index]
        if l == 1:
            X = X[:,0]
        return X
