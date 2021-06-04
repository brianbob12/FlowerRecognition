import numpy as np
import tensorflow as tf

#function that gets an input of any shape and has output shape [None]

class FlattenLayer:

  def __init__(self):
    pass
  
  #input must be numpy
  #returns numpy
  def execute(self,input):
    return tf.flatten(input.flatten())

  def getTrainableVariables(self):
    return []