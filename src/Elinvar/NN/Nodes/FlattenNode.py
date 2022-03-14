import numpy as np
from .Node import Node
from tensorflow import concat
from tensorflow import reshape

#function that gets an input of any shape and has output shape [None]

class FlattenNode(Node):
  
  #TODO count output shape
  def connect(self, connections):
      return super().connect(connections)
  
  #returns numpy
  def execute(self,inputs):
    myInput=concat(inputs,1)
    batchSize=myInput.shape[0]
    newShape=1
    #I think this is ok because it it just uses shapes which aren't stored in VRAM anyway
    for i in myInput.shape[1:]:
      newShape*=i
    return reshape(myInput,shape=[batchSize,newShape])
