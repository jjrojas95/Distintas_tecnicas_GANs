import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

class Maxout(Layer):
    """Layer that applies a basic block of maxout networks proposed by Goodfellow et al

    
    It corresponds to third section of the article "Maxout Networks" available at 
    https://arxiv.org/pdf/1302.4389.pdf summary, appply activation function called
    max out unit, which obtains the maximum value of a series of linear operations:

    .. math:: 
        
        h_i(t) = \max_{j \in [1,k]} z_{ij} \\
        where~z_{ij} = x^T \cdot W_{.., i,j} + b_{ij} 

    Arguments:

        units: Integer, outpout space dimension.
        k: Integer, number of linear equations per unit to evaluate selecting the 
            maximum value, Ian Goodfellow also calls him "num_pieces"
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights 
            matrix.
        bias_regularizer: Regularizer function applied to the bias vector.

    Input shape:
        Tensor with shape: `(batch_size, input_dim)`
    
    Output shape:
        Tensor with shape: `(batch_size, units)`
    """
    def __init__(self, 
                units,k, 
                kernel_initializer='glorot_uniform',
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
        
        self.kernel_shape = self.kernel.shape.as_list()

        self.bias = self.add_weight(
            "bias", 
            shape=[self.units, self.k],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            dtype=self.dtype,
            trainable=True)
        super(Maxout, self).build(input_shape)
    
    def call(self, x):
        z = tf.tensordot(x, self.kernel, [[-1], [0]]) + self.bias
        h = tf.reduce_max(z, axis = -1)
        return h
    
    def get_config(self):
        config = super(Maxout, self).get_config()
        config.update({
            'units': self.units,
            'k': self.k,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config