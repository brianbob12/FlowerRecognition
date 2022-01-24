import tensorflow as tf
import numpy as np

from .Exceptions import *

from .ConvolutionLayer import ConvolutionLayer

class TransposeConvolutionLayer(ConvolutionLayer):

  #override execute
  def execute(self, inputs):
      return tf.nn.conv2d_transpose(inputs,self.filter,self.strides,"VALID")