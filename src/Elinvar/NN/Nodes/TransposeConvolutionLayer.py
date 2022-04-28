from tensorflow.nn import conv2d_transpose
from tensorflow import concat
import numpy as np

from ..Exceptions import *

from .ConvolutionLayer import ConvolutionLayer

class TransposeConvolutionLayer(ConvolutionLayer):

  #override execute
  def execute(self, inputs):
    if not self.built:
      raise(operationWithUnbuiltNode("execute"))
    else:
      myInputs=concat(inputs,-1)
      return conv2d_transpose(myInputs,self.filter,self.strides,"VALID")

  def connect(self,connections):
    #there's a little bit of redundancy here
    super().connect(connections)
    self.outputShape=[
      int(self.inputShape[0]+((self.kernelSize//2)*2)/self.stride),
      int(self.inputShape[1]+((self.kernelSize//2)*2)/self.stride),
      self.numberOfKernels
    ]
  
  def exportNode(self,path,subdir):
    accessPath=super().exportNode(path,subdir)

    #save type
    #NOTE this will be overwritten by children
    #therefore this saves the lowest class of the node
    with open(accessPath+"\\type.txt","w") as f:
      f.write("TransposeConvolutionLayer")

    return accessPath