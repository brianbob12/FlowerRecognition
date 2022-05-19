#used to lookup nodes
from abc import ABCMeta
from typing import Optional

from Elinvar.NN.Exceptions import UnregisteredNode
from . import *

#static
class NodeNameLookup:
  @staticmethod 
  def getNodeFromName(name:str)->ABCMeta:
    if name=="AdaIN":
      return(AdaIN)
    if name=="Buildablenode":
      return(BuildableNode)
    if name=="ConvolutionLayer":
      return(ConvolutionLayer)
    if name=="DenseLayer":
      return(DenseLayer)
    if name=="FlattenNode":
      return(FlattenNode)
    if name=="InputNode":
      return(InputNode)
    if name=="InstanceNormalizationNode":
      return(InstanceNormalizationNode)
    if name=="MaxPoolingNode":
      return(MaxPoolingNode)
    if name=="Node":
      return(Node)
    if name=="TransposeConvolutionLayer":
      return(TransposeConvolutionLayer)
    raise(UnregisteredNode(nodeName=name)) 
  
  @staticmethod
  def getNameFromNode(node:Node)->str:
    #must check from the bottom of the hierarchy
    if isinstance(node,TransposeConvolutionLayer):
      return("TransposeConvolutionLayer")
    if isinstance(node,ConvolutionLayer):
      return("ConvolutionLayer")
    if isinstance(node,AdaIN):
      return("AdaIN")
    if isinstance(node,InstanceNormalizationNode):
      return("InstanceNormalizationNode")
    if isinstance(node,DenseLayer):
      return("DenseLayer")
    if isinstance(node,FlattenNode):
      return("FlattenNode")
    if isinstance(node,MaxPoolingNode):
      return("MaxPoolingNode")
    if isinstance(node,InputNode):
      return("InputNode")

    #this must be below all subclasses of BuildableNode
    if isinstance(node,BuildableNode):
      return("BuildableNode")
    if isinstance(node,Node):
      return("Node")
    
    raise(UnregisteredNode(nodeObject=node)) 

