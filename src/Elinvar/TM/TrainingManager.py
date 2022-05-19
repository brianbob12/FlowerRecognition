#try to make this selective

from typing import Callable, List, Optional
from Elinvar.TM.Modules.Module import Module
from Elinvar.TM.Modules.Conditions import Condition

from .TrainingEpisode import TrainingEpisode

from os import mkdir
#a class to manage training using training episodes

class TrainingManager: 
  def __init__(self):
    self.trainingQue:List[Callable[[],TrainingEpisode]]=[]#list of functions to create training episodes OR a generator
    self.currentTrainingEpisode:Optional[TrainingEpisode]=None

    self.modules:List[Module]=[]

    #set default exit condition to avoid exceptions
    #This should be overwritten to something useful
    self.exitCondition:Condition=Condition()
    self.exitCondition.met=True

  def setEpisodeExitCondition(self,condition:Condition):
    #recursive function to add conditions to self.modules
    def addCondition(c:Condition):
      if not c in self.modules:
        self.modules.append(c)
      for dependency in c.dependencies:
        addCondition(dependency)

    self.exitCondition:Condition=condition
    addCondition(condition)

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
      if self.exitCondition.met:
        running=False

    for module in self.modules:
      module.endOfEpisode(currentTrainingEpisode,self.lastCrossVal)