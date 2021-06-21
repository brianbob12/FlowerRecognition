import numpy as np
import tensorflow as tf

#function that gets an input of any shape and has output shape [None]

class FlattenLayer:

  def __init__(self):
    pass
  
  
  #returns numpy
  def execute(self,inp):
    batchSize=inp.shape[0]
    newShape=1
    for i in inp.shape[1:]:
      newShape*=i
    return tf.reshape(inp,shape=[batchSize,newShape])

  def getTrainableVariables(self):
    return []