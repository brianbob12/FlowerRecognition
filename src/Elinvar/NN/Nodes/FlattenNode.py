
from numpy import prod

from Elinvar.NN.Exceptions import invalidNodeConnection
from .Node import Node
from tensorflow import concat
from tensorflow import reshape

#function that gets an input of any shape and has output shape [None]

class FlattenNode(Node):

  def __init__(self, name=None, protected=False, ID=None):
      super().__init__(name, protected=protected, ID=ID)
      self.outputShape=[0]
      self.imported=False
      self.totalTrainableVariables=0

  #TODO count output shape
  def connect(self, connections):
    tad=0
    for node in connections:
      tad+=prod(node.outputShape)

    if self.imported:
      if(self.outputShape[0]!=tad):
        raise(invalidNodeConnection([tad],self.outputShape))
    else:
      self.outputShape[0]+=tad
    return super().connect(connections)
  
  #returns numpy
  def execute(self,inputs):
    myInput=concat(inputs,1)
    batchSize=myInput.shape[0]

    return reshape(myInput,shape=[batchSize,self.outputShape[0]])

  def importNode(self, myPath: str, subdir: str):
    self.imported=True
    return super().importNode(myPath, subdir)

  def exportNode(self, path:str, subdir:str):
      accessPath= super().exportNode(path, subdir)
      #save type
      #NOTE this will be overwritten by children
      #therefore this saves the lowest class of the node
      with open(accessPath+"\\type.txt","w") as f:
        f.write("FlattenNode")

      return accessPath
