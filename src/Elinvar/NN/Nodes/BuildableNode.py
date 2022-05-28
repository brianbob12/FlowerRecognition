from abc import abstractmethod
from os import access
from typing import List, Optional
from Elinvar.NN.Exceptions import operationWithUnbuiltNode
from .Node import Node 
from tensorflow import Tensor

class BuildableNode(Node):
  def __init__(self,name:Optional[str]=None,protected:bool=False,ID:Optional[int]=None):
    super().__init__(name=name,protected=protected,ID=ID)
    self.built:bool=False

  def getTrainableVariables(self):
    if not self.built:
      raise(operationWithUnbuiltNode(self.ID,"getTrainableVariables"))
    else:
      return super().getTrainableVariables()

  @abstractmethod
  def build(self,seed:Optional[int]=None) -> int:
    self.built=True
    self.totalTrainableVariables=0
    return 0

  def execute(self, inputs:List[Tensor]):
    if not self.built:
      raise(operationWithUnbuiltNode(self.ID,"execute"))
    else:
      return super().execute(inputs)

  def getValue(self):
    if not self.built:
      raise(operationWithUnbuiltNode(self.ID,"getValue"))
    else:
      return super().getValue()

  def exportNode(self, path:str, subdir:str):
    if not self.built:
      raise(operationWithUnbuiltNode(self.ID,"exportNode"))
    
    accessPath= super().exportNode(path, subdir)

    #save type
    #NOTE this will be overwritten by children
    #therefore this saves the lowest class of the node
    with open(accessPath+"\\type.txt","w") as f:
      f.write("BuildableNode")

    return accessPath
  
  def importNode(self, myPath:str, subdir:str):
    accessPath,connections= super().importNode(myPath, subdir) 

    self.built=True

    return accessPath,connections