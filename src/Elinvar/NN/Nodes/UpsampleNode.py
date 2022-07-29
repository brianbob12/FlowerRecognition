from typing import List, Optional, Tuple
from tensorflow import Tensor

from Elinvar.NN.Exceptions import invalidNodeConnection, invalidNumberOfNodeInputs
from .Node import Node
from tensorflow.image import resize
from tensorflow.image import ResizeMethod

class UpsampleNode(Node):
  
  def __init__(self,name:Optional[str]=None,protected:bool=False,ID:Optional[int]=None):
    self.imported:bool=False
    super().__init__(name,protected,ID)

  def new(self,newHeight:int,newWidth:int):
    self.newHeight:int=newHeight
    self.newWidth:int=newWidth

  def execute(self,input:Tensor)->Tensor:
    return resize(input,(self.newHeight,self.newWidth),method=ResizeMethod.BILINEAR)

  def connect(self,connections:List[Node])->None:
    if len(self.inputConnections)==1:
      raise(invalidNumberOfNodeInputs(len(connections),0))
    if len(connections)>1:
      raise(invalidNumberOfNodeInputs(len(connections),1))

    #check the connection has batch + 3 dimensions
    if len(connections[0].outputShape)<3:
      raise(invalidNodeConnection(connections[0].outputShape,[None,None,None]))

    if self.imported:
      #check that the connection meets the expected shape
      if self.outputShape[2]!=connections[0].outputShape[2]:
        raise(invalidNodeConnection(connections[0].outputShape,[None,None,self.outputShape[2]]))
    else: 
      self.outputShape=[self.newHeight,self.newWidth,connections[0].outputShape[2]]

    return super().connect(connections)

  def importNode(self, myPath: str, subdir: str) -> Tuple[str, List[int]]:
    self.imported=True
    self.newHeight=self.outputShape[0]
    self.newWidth=self.outputShape[1]

    return super().importNode(myPath, subdir)

  def exportNode(self, path: str, subdir: str) -> str:
    accessPath= super().exportNode(path, subdir)

    #save type
    #NOTE this will be overwritten by children
    #therefore this saves the lowest class of the node
    with open(accessPath+"\\type.txt","w") as f:
      f.write("UpsampleNode")

    return accessPath