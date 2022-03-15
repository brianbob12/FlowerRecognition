from Elinvar.NN.Exceptions import operationWithUnbuiltNode
from .Node import Node 

class BuildableNode(Node):
  def __init__(self,name=None,protected=False):
    super().__init__(name=name,protected=protected)
    self.built=False

  #made to be overwritten
  def build(self):
    self.built=True

  def execute(self, inputs):
    if not self.built:
      raise(operationWithUnbuiltNode("execute"))
    else:
      return super().execute(inputs)

  def getValue(self):
    if not self.built:
      raise(operationWithUnbuiltNode(self.ID,"getValue"))
    else:
      return super().getValue()