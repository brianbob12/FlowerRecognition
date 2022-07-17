
from typing import List, Optional
from numpy import prod
from tensorflow import Tensor
from tensorflow.nn import dropout
from Elinvar.NN.Exceptions import invalidNodeConnection
from .Node import Node

#function that gets an input of any shape and has output shape [None]

class DropoutNode(Node):
  def __init__(self,rate:float, name:Optional[str]=None, protected:bool=False, ID:Optional[int]=None):
      super().__init__(name, protected=protected, ID=ID)
      self.outputShape:List[int]=[0]
      self.totalTrainableVariables:int=0
      self.rate:float=rate

  def execute(self,inputs:Tensor) -> Tensor:
    return inputs

  def trainingExecute(self, inputs: List[Tensor]) -> Tensor:
      return dropout(inputs,rate=self.rate)

  def importNode(self, myPath: str, subdir: str):
    return super().importNode(myPath, subdir)

  def exportNode(self, path:str, subdir:str):
      accessPath= super().exportNode(path, subdir)
      #save type
      #NOTE this will be overwritten by children
      #therefore this saves the lowest class of the node
      with open(accessPath+"\\type.txt","w") as f:
        f.write("DropoutNode")

      return accessPath

