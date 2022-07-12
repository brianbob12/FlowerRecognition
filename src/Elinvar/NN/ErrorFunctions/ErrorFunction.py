
#a parent class holding error functions

from abc import ABC, abstractmethod
from typing import List
from tensorflow import Tensor

class ErrorFunction(ABC):
  #multiple labels is a boolean
  #true for error functions that run for many labels each iteration
  #false for error functions that only run on one label per iteration
  def __init__(self,multipleLabels:bool):
    self.multipleLabels:bool=multipleLabels

  @staticmethod
  @abstractmethod
  def execute(guess:List[Tensor],y:List[Tensor])->float:
    pass