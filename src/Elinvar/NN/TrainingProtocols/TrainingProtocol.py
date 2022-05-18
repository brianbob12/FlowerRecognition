from abc import ABC, abstractmethod
from typing import Any, List
from tensorflow import constant,Tensor
from tensorflow.keras.optimizers import Optimizer
from ..Nodes import Node

class TrainingProtocol(ABC):
  def __init__(self,learningRate:float,optimizer:Optimizer,requiredOutputNodes:List[Node]):
    self.learningRate:float=learningRate
    self.optimizer:Optimizer=optimizer
    self.requiredOutputNodes:List[Node]=requiredOutputNodes

  @abstractmethod
  def getError(self,networkOutputs:List[Tensor],getErrorArgs:List[Any]):
    return(constant(0))