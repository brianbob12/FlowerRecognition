import numpy as np
import tensorflow as tf

#function that gets an input of any shape and has output shape [None]

class FlattenLayer:

  def __init__(self):
    pass
  
  
  #returns numpy
  def execute(self,inp):
    print(np.array(inp).shape())
    return inp.flatten()

  def getTrainableVariables(self):
    return []