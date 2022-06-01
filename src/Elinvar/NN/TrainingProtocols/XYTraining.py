
from typing import List,Any
from Elinvar.NN.ErrorFunctions.ErrorFunction import ErrorFunction
from Elinvar.NN.Nodes import Node
from Elinvar.NN.TrainingProtocols.TrainingProtocol import TrainingProtocol
from tensorflow import Tensor
from .TrainingProtocol import TrainingProtocol


class XYTraining (TrainingProtocol):
  def __init__(self,
  learningRate:float,
  optimizer:type,
  requiredOutputNodes:List[Node],
  errorFunction:ErrorFunction):
    super().__init__(learningRate,optimizer,requiredOutputNodes)
    self.errorFunction:ErrorFunction=errorFunction
  
  def getError(self,networkOutputs:List[Tensor],getErrorArgs:List[Any])->float:
    Y=getErrorArgs[0]
    return(self.errorFunction.execute(networkOutputs,Y))
