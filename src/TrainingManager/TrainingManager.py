import TrainingEpisode

#a class to manage training using training episodes

class TrainingManager: 
  def __init__(self):
    self.datasets={}
    self.trainingQue=[]#list of trainingEpisode
    #TODO make dataset class
    pass

  #can use one or both
  def setEpisodeEndRequirements(self,maxIterations=None,minErrorDerivative=None):
    self.maxIterations=maxIterations
    self.minErrorDerivative=minErrorDerivative

    if(maxIterations==None):
      self.maxIterationsConstraint=False
    else:
      self.maxIterationsConstraint=True
    if(minErrorDerivative==None):
      self.minErrorDerivativeConstraint=False
    else:
      self.minErrorDerivativeConstraint=True

    if( (not self.minErrorDerivativeConstraint) and ( not self.maxIterationsConstraint)):
      #TODO write this error
      raise()

  def addDataSet(self,datasetName,files,extractionFunction):
    self.datasets[datasetName]={
      "files":files,
      "extract":extractionFunction
    }

  def runSeries(self):
    pass

  def runEpisode(self):
    pass
