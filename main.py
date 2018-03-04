import numpy as np
import libnn
import libgraph

classes = [-1,1]
samplesize = 600
nn = libnn.fullnn(2,6,3,classes)
X = np.random.uniform(low=-5.0,high=5.0,size=(samplesize,2))
Y,_ = nn.predict(X)
libgraph.plot_classes(X=X,Y=Y,classes=classes,portion=0.2)
nn_train = libnn.fullnn(2,12,3,classes)
nn_train.fit(data=X[:400],labels=Y[:400],n_iter=2000)
print('test score: {0:.4f}'.format(nn_train.score(X[400:],Y[400:])))
