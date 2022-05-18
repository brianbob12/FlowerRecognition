#try to make this selective

from typing import Callable, List, Optional
from Elinvar.TM.Modules.Module import Module

from .TrainingEpisode import TrainingEpisode

from os import mkdir
#a class to manage training using training episodes

class TrainingManager: 
  def __init__(self):
    self.trainingQue:List[Callable[[],TrainingEpisode]]=[]#list of functions to create training episodes OR a generator
    self.currentTrainingEpisode:Optional[TrainingEpisode]=None

    self.modules:List[Module]=[]

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
      #Garbage collector should be deleting these once we're done with them
      currentTrainingEpisode=function()
      currentTrainingEpisodeIndex:int=episodeIndex
      self.runEpisode(currentTrainingEpisode,currentTrainingEpisodeIndex)

      try:
        mkdir(saveDirectory)
      except FileExistsError as e:
        pass
      except Exception as e:
        print(e)


    #run modules
    for module in self.modules:
      module.endOfQue()
  

  def runEpisode(self,currentTrainingEpisode:TrainingEpisode,currentTrainingEpisodeIndex:int):
    #run modules
    for module in self.modules:
      module.startOfEpisode(currentTrainingEpisode,currentTrainingEpisodeIndex)

    def crossValCallback(index:int,crossValError:float):
      self.lastCrossVal=crossValError
      for module in self.modules:
        module.endOfCrossVal(currentTrainingEpisode,index,crossValError)

    def iterationCallback(index:int,trainingError:float,iterationTime:float):
      for module in self.modules:
        module.endOfIteration(currentTrainingEpisode,index,trainingError,iterationTime)

    #training variables
    self.lastCrossVal=-1

    running=True

    while running:
      currentTrainingEpisode.train(
        iterationCallback=iterationCallback,
        crossValCallback=crossValCallback,
        )
      #check exit requirements
      if self.maxIterationsConstraint:
        if currentTrainingEpisode.iterationCounter>=self.maxIterations:
          print("MAX iterations hit, exiting")
          running=False

    for module in self.modules:
      module.endOfEpisode(currentTrainingEpisode,self.lastCrossVal)