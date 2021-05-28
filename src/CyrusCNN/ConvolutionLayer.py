import tensorflow as tf
import numpy as np

from CyrusCNN.Exceptions import invalidStrideLength, invalidStrideType

class ConvolutionLayer:
  
  def __init__(self):
    pass

    #filterSize is interger
    #strides is [int,int,int,int]
  def newLayer(self,filterSize,strides):
    weightInitSTDDEV=0.1
    self.filter=tf.Variable(tf.truncated_normal(shape=[filterSize,filterSize,3,3],stddev=weightInitSTDDEV))

    #check if strides is compatable
    if(len(strides)!=4):
      raise(invalidStrideLength(strides))
    else:
      for stride in strides:
        if(int(stride)!=stride):
          #not castable to an int
          raise(invalidStrideType(stride)) 
  
    self.strides=strides
  
  #inputs have shape [None,a,a,3] tf.float32
  def execute(self,inputs):
   return tf.nn.conv2d(inputs,self.filter,self.strides,"VALID") 

  #return a list of the trainable variables
  def getTrainableVariables(self):
    return self.fileter