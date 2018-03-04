import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import tensorflow as tf

class fullnn:
    """
    This class is used to generate a fully L-layer and W-node-per-layer
    connected neural network that can generate and predict binary labels
    and be trained using SGD with minibatch.
    The nonlinear node simply use ReLU. And the loss function uses log
    loss.
    """
    def __init__(self,dim,width,depth,classes):
        self._dim = dim
        self._width = width
        self._depth = depth
        self._classes = classes
        self._n_classes = len(classes)
        self._total_iter = 0
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        self._model_fn()

    @property
    def dim(self):
        return self._dim
    @property
    def width(self):
        return self._width
    @property
    def depth(self):
        return self._depth
    @property
    def classes(self):
        return self._classes
    @property
    def total_iter(self):
        return self._total_iter

    def _model_fn(self):
        with self._graph.as_default():
            global_step = tf.Variable(0,trainable=False,name='global')
            x = tf.placeholder(dtype=tf.float32,
                shape=[None,self._dim],name='features')
            y = tf.placeholder(dtype=tf.uint8,
                shape=[None],name='labels')
            hl = x
            initializer = tf.glorot_uniform_initializer()
            for idx in range(self._depth):
                hl_name = 'Hidden_Layer' + str(idx)
                hl = tf.layers.dense(inputs=hl,units=self._width,
                    kernel_initializer=initializer,
                    activation=tf.nn.relu,
                    name=hl_name)

            logits = tf.layers.dense(inputs=hl,units=self._n_classes,
                name='Logits')
            tf.add_to_collection("Output",logits)
            probabs = tf.nn.softmax(logits)
            tf.add_to_collection("Output",probabs)
            onehot_labels = tf.one_hot(indices=y,depth=self._n_classes)
            log_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels,logits=logits
            )
            tf.add_to_collection('Loss',log_loss)
            self._sess.run(tf.global_variables_initializer())

    def predict(self,data):
        with self._graph.as_default():
            feed_dict = {'features:0':data}
            logits,probabs = tf.get_collection('Output')
            predictions = {
                'indices':tf.argmax(input=logits,axis=1),
                'probabilities':probabs
            }

        results = self._sess.run(predictions,feed_dict=feed_dict)
        classes = [self._classes[index] for index in
            results['indices']]
        probabilities = results['probabilities']
        return classes,probabilities

    def score(self,data,labels):
        predictions,_ = self.predict(data)
        s = 0.
        for idx in range(len(data)):
            s += predictions[idx]==labels[idx]
        accuracy = s / len(data)
        return accuracy

    def fit(self,data,labels,batch_size=1,n_iter=1000):
        indices = [self._classes.index(label) for label in labels]
        indices = np.array(indices)
        with self._graph.as_default():
            loss = tf.get_collection('Loss')[0]
            global_step = self._graph.get_tensor_by_name('global:0')
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=10)
            train_op = optimizer.minimize(loss=loss,
                global_step=global_step)

        for idx in range(n_iter):
            rand_list = np.random.randint(len(data),size=batch_size)
            feed_dict = {'features:0':data[rand_list,:],
                'labels:0':indices[rand_list]}
            if idx % 10 == 1:
                print('iter: {0:d}, loss: {1:.4f}'.format(
                    idx, self._sess.run(loss,feed_dict)))
            self._sess.run(train_op,feed_dict)
            self._total_iter += 1

    def get_params(self,deep=False):
        params = {
            'dim': self._dim,
            'width': self._width,
            'depth': self._depth,
            'classes': self._classes,
        }
        return params

    def __del__(self):
        self._sess.close()
        print('Session is closed.')
