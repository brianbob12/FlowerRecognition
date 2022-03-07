from tf.nn import conv2d_transpose
import numpy as np

from ..Exceptions import *

from .ConvolutionLayer import ConvolutionLayer

class TransposeConvolutionLayer(ConvolutionLayer):

  #override execute
  def execute(self, inputs):
    if not self.built:
      raise(operationWithUnbuiltNode("execute"))
    else:
      return conv2d_transpose(inputs,self.filter,self.strides,"VALID")

  def connect(self,connections):
    super().connect(connections)
    self.outputShape=[
      self.inputShape[0]+((self.kernelSize//2)*2)/self.stride,
      self.inputShape[1]+((self.kernelSize//2)*2)/self.stride,
      self.numberOfKernels
    ]