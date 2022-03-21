
from Elinvar.NN.TrainingProtocols.TrainingProtocol import TrainingProtocol

from .TrainingProtocol import TrainingProtocol


class XYTraining (TrainingProtocol):
  def __init__(self,
  learningRate,
  optimizer,
  requiredOutputNodes,
  errorFunction):
    super().__init__(learningRate,optimizer,requiredOutputNodes)
    self.errorFunction=errorFunction
  
  def getError(self,networkOutputs,Y):
    return(self.errorFunction.execute(networkOutputs,Y))
