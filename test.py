import numpy as np
import libgraph, libnn

m_max = 10000
m = 7000
classes = [-1,1]
with open('data/ideal_Xtest.txt') as f:
    X = np.loadtxt(f,delimiter=' ')
with open('data/ideal_Ytest.txt') as f:
    Y = np.loadtxt(f,delimiter=' ')

Xtrain = X[:m]
Ytrain = Y[:m]
Xtest = X[m:m_max]
Ytest = Y[m:m_max]

clf = libnn.fullnn(2,10,1,classes)
clf.fit(batch_size=100,data=Xtrain,labels=Ytrain,n_iter=10000)
print('test score: {0:.4f}'.format(clf.score(Xtest,Ytest)))
