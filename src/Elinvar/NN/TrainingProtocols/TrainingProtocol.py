class TrainingProtocol:
  def __init__(self,learningRate,optimizer,requiredOutputNodes):
    self.learningRate=learningRate
    self.optimizer=optimizer
    self.requiredOutputNodes=requiredOutputNodes

  def getError(self,networkOutputs):
    return(0)