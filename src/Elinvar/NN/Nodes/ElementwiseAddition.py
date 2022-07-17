
from copy import copy
from typing import List, Optional
from tensorflow.math import add_n
from tensorflow import Tensor
from Elinvar.NN.Exceptions import invalidNodeConnection
from .Node import Node

#Node to add tensors elementwise
class ElementwiseAddition(Node):

  def __init__(self,name:Optional[str]=None,protected:bool=False,ID:Optional[int]=None):
    super().__init__(name,protected,ID)

  def connect(self, connections: List[Node]):
    if len(connections)==0:
      return

    if len(self.inputConnections)==0:
      #check connections have the same shape
      #take copy
      self.outputShape=copy(connections[0].outputShape)

    if len(connections)>1:
      for connection in connections:
        if connection.outputShape!=self.outputShape:
          raise(invalidNodeConnection(connection.outputShape,self.outputShape))


    return super().connect(connections)

  def execute(self,inputs:List[Tensor])->Tensor:
    # add elements elementwise
    return add_n(inputs)

  def exportNode(self, path, subdir):
    accessPath= super().exportNode(path, subdir)

    #save type
    #NOTE this will be overwritten by children
    #therefore this saves the lowest class of the node
    with open(accessPath+"\\type.txt","w") as f:
      f.write("InstanceNormalizationNode") 
