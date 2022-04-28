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

  def exportNode(self, path:str, subdir:str):
    accessPath= super().exportNode(path, subdir)
    #save type
    #NOTE this will be overwritten by children
    #therefore this saves the lowest class of the node
    with open(accessPath+"\\type.txt","w") as f:
      f.write("InputNode")


    return accessPath