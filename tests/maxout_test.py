import unittest
from unittest import TestCase
import tensorflow as tf
from tensorflow.keras.layers import Layer

from helper_func.custom_layers import Maxout

class TestMaxoutLayer(TestCase):
    def setUp(self):
        self.k = 5
        self.dim = 240
        self.maxout_instance = Maxout(input_shape = (784,))

    def test_maxout_type(self):
        self.assertTrue(isinstance(self.maxout_instance, Layer))
    
    def test_kernel_shape(self):
        shape = self.maxout_instance.kernel.shape.as_list()
        self.assertEqual(shape, (None, input_shape[0], self.dim, self.k))
    
    def test_bias_shape(self):        
        shape = self.maxout_instance.bias.shape.as_list()
        self.assertEqual(shape, (None, self.dim, self.k))
    
    def test_output_shape(self):        
        self.assertEqual(self.maxout_instance.output_shape(), (None, self.dim))
    
    def test_output_shape(self):      
        self.assertEqual(self.maxout_instance.output_shape(), (None, self.dim))

if __name__ == '__main__':
    unittest.main()