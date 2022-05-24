from abc import ABC, abstractmethod
from typing import Any, List
from tensorflow import constant,Tensor
from ..Nodes import Node

class TrainingProtocol(ABC):
  def __init__(self,learningRate:float,optimizer:type,requiredOutputNodes:List[Node]):
    self.learningRate:float=learningRate
    self.optimizer:type=optimizer
    self.requiredOutputNodes:List[Node]=requiredOutputNodes

  @abstractmethod
  def getError(self,networkOutputs:List[Tensor],getErrorArgs:List[Any])->Tensor:
    return(constant(0))