import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
m = 20
m_max = 1000

Xtrain = Xtrain[:m]
Ytrain = Ytrain[:m]
# np.random.shuffle(Ytrain)
Xtest = Xtest[m:m_max]
Ytest = Ytest[m:m_max]
# np.random.shuffle(Ytest)
libgraph.plot_classes(Xtest,Ytest,classes,1)
# TITLE = 'random_labeled_'
TITLE = 'correct_labeled_'

clf1 = libnn.fullnn(2,20,4,classes)
lips_list = []
excess_err = []
total_epoch = 2000
round_list = np.linspace(1,np.log10(total_epoch),12)
last_epoch = 0
for idx in list(map(int,10**round_list)):
    n_epoch = idx - last_epoch
    last_epoch = idx
    clf1.fit(batch_size=5,data=Xtrain,labels=Ytrain,n_epoch=n_epoch)
    train_score = clf1.score(Xtrain,Ytrain)
    test_score = clf1.score(Xtest,Ytest)
    print('train score: {0:.4f},\ntest score: {1:.4f},\nexcess error: {2:.4f}'.format(train_score,
        test_score,train_score - test_score))
    excess_err.append(train_score - test_score)
    lips_c = 1
    for key in clf1.trainable_params:
        if 'kernel' in key:
            lips_c = lips_c * np.linalg.norm(clf1.trainable_params[key],2)
    lips_list.append(lips_c)
fig1 = plt.figure()
plt.plot(round_list,excess_err,'b^-')
plt.title(TITLE+' excess error')
plt.savefig('image/'+TITLE+'excess_err.eps')
plt.close(fig1)
fig2 = plt.figure()
plt.plot(round_list,lips_list,'bo-')
plt.title(TITLE+' Lipschitz constant')
plt.savefig('image/'+TITLE+'lipschitz.eps')
plt.close(fig2)
fig3 = plt.figure()
n = len(classes)
colors = cm.rainbow(np.linspace(0,1,n))
c = list()
Ytest,_ = clf1.predict(Xtest)
for idx in range(len(Ytest)):
    for jdx in range(n):
        if Ytest[idx]==classes[jdx]:
            c.append(colors[jdx])
plt.scatter(Xtest[:,0],Xtest[:,1],marker='o',facecolors='none',edgecolors=c)
c = list()
for idx in range(len(Ytrain)):
    for jdx in range(n):
        if Ytrain[idx]==classes[jdx]:
            c.append(colors[jdx])
plt.scatter(Xtrain[:,0],Xtrain[:,1],c=c,marker='x')
plt.axis('equal')
plt.savefig('image/'+TITLE+'illustration.eps')
plt.close(fig3)



# clf2 = libnn.fullnn(2,50,4,classes)
# np.random.shuffle(Ytrain)
# np.random.shuffle(Ytest)
# clf2.fit(batch_size=100,data=Xtrain,labels=Ytrain,n_iter=10000)
# print('test score: {0:.4f}'.format(clf2.score(Xtest,Ytest)))
