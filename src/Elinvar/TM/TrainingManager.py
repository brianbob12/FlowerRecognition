#try to make this selective

from Elinvar.TM.Modules.Module import Module

from .TrainingEpisode import TrainingEpisode

from os import mkdir
#a class to manage training using training episodes

class TrainingManager: 
  def __init__(self):
    self.trainingQue=[]#list of lambda functions to create trainng episodes OR a generator
    self.currentTrainingEpisode=None

    self.modules:list[Module]=[]

  #can use one or both
  #TODO set minimIterations
  def setEpisodeEndRequirements(self,maxIterations:int=None ,minCrossVal:float=None):
    self.maxIterations=maxIterations
    self.minCrossVal=minCrossVal

    if(maxIterations==None):
      self.maxIterationsConstraint=False
    else:
      self.maxIterationsConstraint=True
    if(minCrossVal==None):
      self.minErrorConstraint=False
    else:
      self.minErrorConstraint=True

    if( (not self.minErrorConstraint) and ( not self.maxIterationsConstraint)):
      #TODO write this error
      raise()

  #episodeCallback has args: TrainingEpisode
  def runQue(self,saveDirectory=".\\runs"):
    #run modules
    for module in self.modules:
      module.startOfQue(saveDirectory)
    
    for episodeIndex,function in enumerate(self.trainingQue):
      #Garbage collector should be deleteing these once we're done with them
      self.currentTrainingEpisode=function()
      self.currentTrainingEpisodeIndex=episodeIndex
      finalCrossValError = self.runEpisode()

      try:
        mkdir(saveDirectory)
      except FileExistsError as e:
        pass
      except Exception as e:
        print(e)

      #export data
      
      self.currentTrainingEpisode.exportData(saveDirectory)

    #run modules
    for module in self.modules:
      module.endOfQue()
  

  def runEpisode(self):
    #run modules
    for module in self.modules:
      module.startOfEpisode(self.currentTrainingEpisode,self.currentTrainingEpisodeIndex)

    def crossValCallback(index:int,crossValError:float):
      self.lastCrossVal=crossValError
      for module in self.modules:
        module.endOfCrossVal(self.currentTrainingEpisode,index,crossValError)

    def iterationCallback(index:int,trainingError:float,iterationTime:float):
      for module in self.modules:
        module.endOfIteration(self.currentTrainingEpisode,index,trainingError,iterationTime)

    #training variables
    self.lastCrossVal=-1

    running=True

    while running:
      self.currentTrainingEpisode.train(
        iterationCallback=iterationCallback,
        crossValCallback=crossValCallback,
        )
      #check exit requirements
      if self.maxIterationsConstraint:
        if self.currentTrainingEpisode.iterationCounter>=self.maxIterations:
          print("MAX iterations hit, exiting")
          running=False
      elif self.minErrorDerivativeConstraint:
        if self.lastCrossValDerivativeEstimation!=-1 and self.lastCrossValDerivativeEstimation<self.minErrorDerivative:
          print("MIN error derrivate hit, exiting")
          running=False

    for module in self.modules:
      module.endOfEpisode(self.currentTrainingEpisode,self.lastCrossVal)
    return self.lastCrossVal