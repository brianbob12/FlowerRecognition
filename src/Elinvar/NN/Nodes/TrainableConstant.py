
from typing import Optional
from .Node import Node

class TrainableConstant(Node):
  def __init__(self,name:Optional[str]=None,protected:bool=False,ID:Optional[int]=None):
    super().__init__(name,protected,ID)