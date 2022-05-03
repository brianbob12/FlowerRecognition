from os import access
from Elinvar.NN.Exceptions import operationWithUnbuiltNode
from .Node import Node 

class BuildableNode(Node):
  def __init__(self,name=None,protected=False,ID=None):
    super().__init__(name=name,protected=protected,ID=ID)
    self.built=False

  def getTrainableVariables(self):
    if not self.built:
      raise(operationWithUnbuiltNode(self.ID,"getTrainableVariables"))
    else:
      return super().getTrainableVariables()

  #made to be overwritten
  def build(self) -> int:
    self.built=True
    self.totalTrainableVariables=0
    return 0

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

  def exportNode(self, path, subdir):
    if not self.built:
      raise(operationWithUnbuiltNode(self.ID,"exportNode"))
    
    accessPath= super().exportNode(path, subdir)

    #save type
    #NOTE this will be overwritten by children
    #therefore this saves the lowest class of the node
    with open(accessPath+"\\type.txt","w") as f:
      f.write("BuildableNode")

    return accessPath
  
  def importNode(self, myPath, subdir):
    accessPath,connections= super().importNode(myPath, subdir) 

    self.built=True

    return accessPath,connections