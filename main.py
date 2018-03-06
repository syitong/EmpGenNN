import numpy as np
import libnn, libgraph, libdata
import matplotlib.pyplot as plt

classes = [-1,1]
samplesize = 1000
nn = libnn.fullnn(2,10,7,classes)
X = np.random.uniform(low=-5.0,high=5.0,size=(samplesize,2))
Y,_ = nn.predict(X)
# X,Y = libdata.unit_circle_ideal(0.1,1,samplesize)
# np.random.shuffle(Y)
libgraph.plot_classes(X=X,Y=Y,classes=classes,portion=0.3)
nn_train = libnn.fullnn(2,10,1,classes)
nn_train.fit(batch_size=10,data=X[:700],labels=Y[:700],n_iter=5000)
print('test score: {0:.4f}'.format(nn_train.score(X[700:],Y[700:])))

# plot the decision boundary of the classifier.
# x = np.linspace(-2,2,500)
# y = np.linspace(-2,2,500)
# xx,yy = np.meshgrid(x,y)
# xy = np.c_[xx.ravel(),yy.ravel()]
# z,_ = nn_train.predict(xy)
# z = np.reshape(z,(500,500))
# fig = plt.figure()
# plt.contour(xx,yy,z,levels=[0])
# plt.axis('equal')
# plt.show()
