import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tf_type = tf.float32

def quadrl(degree):
        if degree == 3:
            nd = tf.constant([1.73205080756888, 0, -1.73205080756888], dtype=tf_type)
            w = tf.constant([0.16666666666667, 0.66666666666667, 0.16666666666667], dtype=tf_type)
            return [nd, w]
        elif degree == 5:
            nd = tf.constant([2.85697001387281, 1.35562617997427, 0, -1.35562617997427, -2.85697001387281], dtype=tf_type)
            w = tf.constant([0.01125741132772, 0.22207592200561, 0.53333333333334, 0.22207592200561, 0.01125741132772], dtype=tf_type)
            return [nd, w]
        elif degree == 7:
            nd = tf.constant([3.75043971772574, 2.36675941073454, 1.15440539473997, 0, -1.15440539473997, -2.36675941073454, -3.75043971772574], dtype=tf_type)
            w = tf.constant([0.00054826885597, 0.03075712396759, 0.24012317860501, 0.45714285714286, 0.24012317860501, 0.03075712396759, 0.00054826885597], dtype=tf_type)
            return [nd, w]  
        elif degree == 9:
            nd = tf.constant([4.51274586339978, 3.20542900285647, 2.07684797867783, 1.02325566378913, 0, -1.02325566378913, -2.07684797867783, -3.20542900285647, -4.51274586339978], dtype=tf_type)
            w = tf.constant([0.00002234584401, 0.00278914132123, 0.04991640676522, 0.24409750289494, 0.40634920634921, 0.24409750289494, 0.04991640676522, 0.00278914132123, 0.00002234584401], dtype=tf_type)            
            return [nd, w] 
    
    
def batch_tf_quad(f, batch_mu, batch_v, degree):
        #\int N(x|mu, v) f(x) \d x 

        [nd,w] = quadrl(degree)
        Ns = tf.size(batch_mu)

        nd = tf.reshape(nd, [1,-1])
        nd = tf.repeat(nd, Ns, axis=0)

        nodes = nd * tf.sqrt(tf.reshape(batch_v, [-1,1])) + tf.reshape(batch_mu, [-1,1])
        vec_nodes = tf.reshape(nodes, [-1,1])

        evals1, evals2 = f(vec_nodes)
        
        evals1 = tf.reshape(evals1, tf.shape(nodes))
        evals2 = tf.reshape(evals2, tf.shape(nodes))

        return tf.matmul(evals1, tf.reshape(w, [-1,1])), tf.matmul(evals2, tf.reshape(w, [-1,1]))

def tf_quad(f, mu, v, degree):
    #\int N(x|mu, v) f(x) \d x  
    
    [nd,w] = quadrl(degree)
    nodes = nd * tf.sqrt(v) + mu
    
    evals1, evals2 = f(nodes)
    
    return tf.reduce_sum(evals1*w), tf.reduce_sum(evals2*w)

class EncodeNN:
    """Feature Encoding NN"""
    # init 0 xavier
    # init 1 msra
    def __init__(self, layers, init='xavier', activation='tanh'):
        self.num_layers = len(layers)
        self.layers = layers
        
        if activation == 'tanh':
            self.activation = tf.tanh
        elif activation == 'relu':
            self.activation = tf.nn.relu
        elif activation == 'sigmoid':
            self.activation = tf.sigmoid
        elif activation == 'leaky_relu':
            self.activation = tf.nn.leaky_relu

        if init == 'xavier':
            self.init = self.xavier_init
        else:
            self.init = self.msra_init

        self.weights = []
        self.biases = []
        for l in range(self.num_layers - 1):
            W = self.init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf_type))
            self.weights.append(W)
            self.biases.append(b)
            
    def forward(self, X):
        H = X
        for l in range(self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            H = self.activation(tf.add(tf.matmul(H, W), b))

        return H

        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev))

    def msra_init(self,size):
        in_dim = size[0]
        out_dim = size[1]    
        msra_stddev = np.sqrt(2.0/(in_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=msra_stddev))
    
class CopyEncodeNN:
    """Feature Encoding NN"""
    # init 0 xavier
    # init 1 msra
    def __init__(self, layers, activation, weights, biases):
        
        self.num_layers = len(layers)
        
        if activation == 'tanh':
            self.activation = tf.tanh
        elif activation == 'relu':
            self.activation = tf.nn.relu
        elif activation == 'sigmoid':
            self.activation = tf.sigmoid
        elif activation == 'leaky_relu':
            self.activation = tf.nn.leaky_relu

        self.weights = []
        self.biases = []
        for l in range(self.num_layers - 1):
            self.weights.append(tf.convert_to_tensor(tf.constant(weights[l]), dtype=tf_type))
            self.biases.append(tf.convert_to_tensor(tf.constant(biases[l]), dtype=tf_type))
            
    def forward(self, X):
        H = X
        for l in range(self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            H = self.activation(tf.add(tf.matmul(H, W), b))

        return H
    
    