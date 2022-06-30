from typing import Callable, List,Optional
from tensorflow import Tensor
from .Node import Node
from Elinvar.NN.Exceptions import nodeNotSetup,missingFileForImport,invalidDataInFile


class InputNode(Node):
  def __init__(self, name=None, protected=False,ID=None):
    self.onExecute:Optional[Callable[[],Tensor]]=None
    super().__init__(name=name, protected=protected,ID=ID)
    self.totalTrainableVariables:int=0

  def setup(self,outputShape:List[int]):
    self.outputShape:List[int]=outputShape


  def execute(self,inputs):
    if self.onExecute==None:
      #the onExecute function must be defined before executing node
      raise(nodeNotSetup("execute"))
    else:
      return self.onExecute() 

  def exportNode(self, path:str, subdir:str) -> str:
    accessPath= super().exportNode(path, subdir)
    #save type
    #NOTE this will be overwritten by children
    #therefore this saves the lowest class of the node
    with open(accessPath+"\\type.txt","w") as f:
      f.write("InputNode")

    return accessPath

  def inputNode(self,myPath:str,subdir:str):
    accessPath,connections = super().importNode(myPath,subdir)

    return accessPath,connections