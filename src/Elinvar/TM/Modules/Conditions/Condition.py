from typing import Callable, List
from ..Module import Module


class Condition(Module):
  __slots__=("met","onChange","dependencies")

  def __init__(self):
    self.met:bool=False
    self.onChange:List[Callable[[bool],None]]=[]
    self.dependencies:List[Condition]=[]

  def setValue(self,value:bool):
    if (value!=self.met):
      self.met=value
      #run callbacks
      for callback in self.onChange:
        callback(value)
    else:
      self.met=value