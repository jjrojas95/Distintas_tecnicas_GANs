import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

class Maxout(Layer):
    def __init__(self, 
                units,k, kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None, 
                **kwargs):

        super(Maxout, self).__init__(**kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        self.k = int(k) if not isinstance(k, int) else k
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", 
            shape=[int(input_shape[-1]), self.units, self.k],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True)

        self.bias = self.add_weight(
            "bias", 
            shape=[self.units, self.k],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            dtype=self.dtype,
            trainable=True)
    
    def call(self, x):
        z = tf.tensordot(x, self.kernel, [[-1], [0]]) + self.bias
        h = tf.reduce_max(z, axis = -1)
        return h