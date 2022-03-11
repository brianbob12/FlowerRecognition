from .Node import Node
from Elinvar.NN.Exceptions import nodeNotSetup


class InputNode(Node):
  def __init__(self, name=None, protected=False):
    self.onExecute=None
    super().__init__(name, protected)

  def setup(self,function,outputShape):
    self.onExecute=function
    self.outputShape=outputShape


  def execute(self, inputs):
    if self.onExecute==None:
      raise(nodeNotSetup("execute"))
    else:
      return self.onExecute() 
