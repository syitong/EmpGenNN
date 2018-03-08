import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_classes(X,Y,classes,portion):
    m = int(len(X) * portion)
    n = len(classes)
    A = np.array(X[0:m])
    Y = Y[0:m]
    colors = cm.rainbow(np.linspace(0,1,n))
    c = list()
    for idx in range(m):
        for jdx in range(n):
            if Y[idx]==classes[jdx]:
                c.append(colors[jdx])
    fig = plt.figure()
    plt.scatter(A[:,0],A[:,1],c=c)
    plt.axis('equal')
    plt.savefig('image/graph.eps')
    plt.close(fig)
    return 1

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
