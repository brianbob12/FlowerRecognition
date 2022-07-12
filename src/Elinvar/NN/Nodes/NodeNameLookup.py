#used to lookup nodes
from abc import ABCMeta

from Elinvar.NN.Exceptions import UnregisteredNode
from . import *

#static
class NodeNameLookup:
  @staticmethod 
  def getNodeFromName(name:str)->ABCMeta:
    if name=="BuildableNode":
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
    if name=="DropoutNode":
      return(DropoutNode)
    if name == "AdaINSetStyle":
      return(AdaINSetStyle)
    if name=="AdaINStyleTransfer":
      return(AdaINStyleTransfer)
    if name=="ElementwiseAddition":
      return(ElementwiseAddition)
    if name=="ScalingLayer":
      return(ScalingLayer)
    if name=="TrainableConstant":
      return(TrainableConstant)
    if name=="UpsampeNode":
      return(UpsampleNode)
    raise(UnregisteredNode(nodeName=name)) 
  
  @staticmethod
  def getNameFromNode(node:Node)->str:
    #must check from the bottom of the hierarchy
    if isinstance(node,TransposeConvolutionLayer):
      return("TransposeConvolutionLayer")
    if isinstance(node,ConvolutionLayer):
      return("ConvolutionLayer")
    if isinstance(node,AdaINStyleTransfer):
      return("AdaINStyleTransfer")
    if isinstance(node,AdaINSetStyle):
      return("AdaINSetStyle")
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
    if isinstance(node,DropoutNode):
      return("DropoutNode")
    if isinstance(node,ElementwiseAddition):
      return("ElementwiseAddition")
    if isinstance(node,ScalingLayer):
      return("ScalingLayer")
    if isinstance(node,TrainableConstant):
      return("TrainableConstant")
    if isinstance(node,UpsampleNode):
      return("UpsampleNode")

    #this must be below all subclasses of BuildableNode
    if isinstance(node,BuildableNode):
      return("BuildableNode")
    if isinstance(node,Node):
      return("Node")
    
    raise(UnregisteredNode(nodeObject=node)) 

