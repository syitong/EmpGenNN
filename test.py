import numpy as np
import libdata, libgraph, libnn

classes = [-1,1]
with open('data/ideal_Xtest.txt') as f:
    Xtest = np.loadtxt(f,delimiter=' ')
with open('data/ideal_Ytest.txt') as f:
    Ytest = np.loadtxt(f,delimiter=' ')
with open('data/ideal_Xtrain.txt') as f:
    Xtrain = np.loadtxt(f,delimiter=' ')
with open('data/ideal_Ytrain.txt') as f:
    Ytrain = np.loadtxt(f,delimiter=' ')
m = 200
m_max = 300

Xtrain = Xtrain[:m]
Ytrain = Ytrain[:m]
Xtest = Xtest[m:m_max]
Ytest = Ytest[m:m_max]

clf1 = libnn.fullnn(2,20,1,classes)
clf1.fit(batch_size=100,data=Xtrain,labels=Ytrain,n_epoch=5)
train_score = clf1.score(Xtrain,Ytrain)
test_score = clf1.score(Xtest,Ytest)
print('''train score: {0:.4f},\n
test score: {1:.4f},\n
excess error: {2:.4f}'''.format(train_score,
    test_score,train_score - test_score))
lips_c = 1
for key in clf1.trainable_params:
    if 'kernel' in key:
        lips_c = lips_c * np.linalg.norm(clf1.trainable_params[key],2)
print(lips_c)
# clf2 = libnn.fullnn(2,50,4,classes)
# np.random.shuffle(Ytrain)
# np.random.shuffle(Ytest)
# clf2.fit(batch_size=100,data=Xtrain,labels=Ytrain,n_iter=10000)
# print('test score: {0:.4f}'.format(clf2.score(Xtest,Ytest)))
