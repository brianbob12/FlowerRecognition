import tensorflow as tf
import numpy as np

#non-trainable layer#
#maxpooling
class PoolingLayer:
  def __init__(self,size,stride):
    self.size=size
    self.stride=stride

  #input has shape[None,a,a,3]
  def execute(self,inputs):
    #out=[]
    #for i in inputs:
    #  out.append(tf.nn.max_pool(inputs,self.size,self.stride,"VALID"))
    #return out
    return tf.nn.max_pool2d(inputs,self.size,self.stride,"VALID")

  #has no trainable variables
  #this function is here so that all layers can hava a .getTrainableVariables
  def getTrainableVariables(self):
    return []

  def exportLayer(superdir,subdir):
    pass

  def importLayer(superdir,subdir):
    pass