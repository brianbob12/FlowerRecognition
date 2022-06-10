#Adaptive Instance Normalization Layer
import imp
from .Node import Node

class AdaINStyleTransfer(Node):

  def connect(self, connections):
    return super().connect(connections)

  def execute(self, inputs):
    return super().execute(inputs)

  def exportNode(self, path, subdir):
    accessPath= super().exportNode(path, subdir)
    #save type
    #NOTE this will be overwritten by children
    #therefore this saves the lowest class of the node
    with open(accessPath+"\\type.txt","w") as f:
      f.write("AdaINStyleTransfer")
    return super().exportNode(path, subdir)