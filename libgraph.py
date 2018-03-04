import numpy as np
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
