
from distutils.log import error
from msilib.schema import Error
from typing import Any, List
from Elinvar.NN.ErrorFunctions.ErrorFunction import ErrorFunction
from Elinvar.NN.Nodes import Node
from Elinvar.NN.TrainingProtocols.TrainingProtocol import TrainingProtocol
from Elinvar.NN import Network
from tensorflow import Tensor

class DiscriminatorNetwork(TrainingProtocol):
  def __init__(
    self,
    learningRate: float,
    optimizer: type,
    requiredOutputNodes: List[Node],
    discriminatorNetwork:Network,
    discriminatorNetworkOutputNodes:List[Node],
    errorFunction:ErrorFunction
    ):
      self.discriminatorNetwork:Network=discriminatorNetwork
      self.discriminatorNetworkOutputNodes:List[Node]=discriminatorNetworkOutputNodes
      self.errorFunction:ErrorFunction=errorFunction
      super().__init__(learningRate, optimizer, requiredOutputNodes)

  def getError(self,networkOutputs,getErrorArgs):
    Y=getErrorArgs[0]
    return(self.errorFunction.execute(networkOutputs,Y))