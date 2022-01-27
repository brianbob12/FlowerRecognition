

#a class to manage training using training episodes

class TrainingManager: 
  def __init__(self):
    self.datasets={}
    #TODO make dataset class
    pass

  def addDataSet(self,datasetName,files,extractionFunction):
    self.datasets[datasetName]={
      "files":files,
      "extract":extractionFunction
    }

  def runSeries(self):
    pass

  def runEpisode(self):
    pass
