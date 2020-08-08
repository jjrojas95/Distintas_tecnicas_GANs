import unittest
from unittest import TestCase
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


from helper_func.custom_layers import Maxout

class TestMaxoutLayer(TestCase):
    def setUp(self):
        self.k = 5
        self.units = 240
        self.input_shape = (784,)
        self.maxout_instance = Maxout(self.units, self.k)
        self.maxout_instance.build(self.input_shape)

    # @unittest.skip("probando el resultado")
    def test_maxout_type(self):
        self.assertTrue(isinstance(self.maxout_instance, Layer))
    
    # @unittest.skip("probando el resultado")
    def test_kernel_shape(self):
        shape = self.maxout_instance.kernel.shape
        self.assertEqual(shape, (self.input_shape[0], self.units, self.k))
    
    # @unittest.skip("probando el resultado")
    def test_bias_shape(self):        
        shape = self.maxout_instance.bias.shape
        self.assertEqual(shape, (self.units, self.k))
    
    # @unittest.skip("probando el resultado")
    def test_output_shape(self):
        input_shape = (5,)
        units = 10
        k = 3
        other_maxout = Maxout(units, k, input_shape=input_shape)
        x = tf.ones((10,5))
        output = other_maxout(x) 

        self.assertEqual(output.shape, (10, units))
    
    # @unittest.skip("probando el resultado")
    def test_output_2samples_1unit(self):
        # Primera parte, una sola unidad dos ejemplos y reducción de k.
        m = 2
        input_shape = (2,)
        units = 1
        k = 3
        w = tf.reshape(tf.cast(tf.range(1,7), tf.float32), (input_shape[-1], units, k))
        other_maxout = Maxout(units, k)
        other_maxout.build((m, *input_shape))
        other_maxout.kernel.assign(w)
        x = tf.reshape(tf.cast(tf.range(1,5), tf.float32), (m, *input_shape))
        expect_val = tf.reshape(tf.convert_to_tensor([15., 33.]), (m, units))
        self.assertEqual(other_maxout(x).numpy().tolist(), expect_val.numpy().tolist())

    def test_output_1samples_2unit(self):
        m = 1
        input_shape = (2,)
        units = 2
        k = 3
        w = tf.reshape(tf.cast(tf.range(1,13), tf.float32), (input_shape[-1], units, k))
        other_maxout = Maxout(units, k)
        other_maxout.build((m, *input_shape))
        other_maxout.kernel.assign(w)
        x = tf.reshape(tf.cast(tf.range(1,3), tf.float32), (m, *input_shape))
        expect_val = tf.reshape(tf.convert_to_tensor([21., 30.]), (m, units))
        self.assertEqual(other_maxout(x).numpy().tolist(), expect_val.numpy().tolist())


if __name__ == '__main__':
    unittest.main()