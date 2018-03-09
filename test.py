import numpy as np
import libdata, libgraph, libnn

X,Y = libdata.unit_circle_ideal(0.2,1,10000)
Xtrain = X[:7000]
Ytrain = Y[:7000]
Xtest = X[7000:]
Ytest = Y[7000:]
np.savetxt('data/ideal_Xtrain.txt',Xtrain)
np.savetxt('data/ideal_Ytrain.txt',Ytrain)
np.savetxt('data/ideal_Xtest.txt',Xtest)
np.savetxt('data/ideal_Ytest.txt',Ytest)
# m_max = 10000
# m = 7000
# classes = [-1,1]
# with open('data/ideal_Xtest.txt') as f:
#     X = np.loadtxt(f,delimiter=' ')
# with open('data/ideal_Ytest.txt') as f:
#     Y = np.loadtxt(f,delimiter=' ')
#
# Xtrain = X[:m]
# Ytrain = Y[:m]
# Xtest = X[m:m_max]
# Ytest = Y[m:m_max]
#
# clf = libnn.fullnn(2,10,1,classes)
# clf.fit(batch_size=100,data=Xtrain,labels=Ytrain,n_iter=10000)
# print('test score: {0:.4f}'.format(clf.score(Xtest,Ytest)))
