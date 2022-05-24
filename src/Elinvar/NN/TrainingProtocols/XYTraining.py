
from typing import List
from Elinvar.NN.Nodes import Node
from Elinvar.NN.TrainingProtocols.TrainingProtocol import TrainingProtocol

from .TrainingProtocol import TrainingProtocol


class XYTraining (TrainingProtocol):
  def __init__(self,
  learningRate:float,
  optimizer,
  requiredOutputNodes:List[Node],
  errorFunction):
    super().__init__(learningRate,optimizer,requiredOutputNodes)
    self.errorFunction=errorFunction
  
  def getError(self,networkOutputs,getErrorArgs):
    Y=getErrorArgs[0]
    return(self.errorFunction.execute(networkOutputs,Y))
