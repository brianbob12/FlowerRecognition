
from distutils.log import error
from msilib.schema import Error
from typing import Any, List,Dict
from Elinvar.NN.ErrorFunctions.ErrorFunction import ErrorFunction
from Elinvar.NN.Nodes import InputNode, Node
from Elinvar.NN.TrainingProtocols.TrainingProtocol import TrainingProtocol
from Elinvar.NN import Network
from tensorflow import Tensor

class DiscriminatorNetworkProtocol(TrainingProtocol):
  def __init__(
    self,
    learningRate: float,
    optimizer: type,
    requiredOutputNodes: List[Node],
    discriminatorNetwork:Network,
    discriminatorNetworkOutputNodes:List[Node],
    outputToInputMap:Dict[Node,InputNode],
    errorFunction:ErrorFunction
    ):
      self.discriminatorNetwork:Network=discriminatorNetwork
      self.discriminatorNetworkOutputNodes:List[Node]=discriminatorNetworkOutputNodes
      self.errorFunction:ErrorFunction=errorFunction
      self.outputToInputMap:Dict[Node,InputNode]=outputToInputMap
      super().__init__(learningRate, optimizer, requiredOutputNodes)

  def getError(self,networkOutputs:List[Tensor],getErrorArgs):
    inputsForDiscriminatorNetwork:Dict[int,Tensor]={}

    for index,value in enumerate(networkOutputs):
      outputNode:Node=self.requiredOutputNodes[index]
      inputNode:InputNode=self.outputToInputMap[outputNode]
      inputsForDiscriminatorNetwork[inputNode.ID]=value 

    self.discriminatorNetwork.clear()
    discriminatorOutputs:List[Tensor]=self.discriminatorNetwork.execute(
      inputsForDiscriminatorNetwork,
      self.discriminatorNetworkOutputNodes)
    Y=getErrorArgs[0]
    return(self.errorFunction.execute(discriminatorOutputs,Y))