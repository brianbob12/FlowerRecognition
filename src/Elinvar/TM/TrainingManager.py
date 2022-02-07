from random import triangular
from typing import final

from numpy import cross, save
from .TrainingEpisode import TrainingEpisode

from os import mkdir
#a class to manage training using training episodes

class TrainingManager: 
  def __init__(self):
    self.datasets={}
    self.trainingQue=[]#list of lambda functions to create trainng episodes OR a generator
    self.currentTrainingEpisode=None
    self.exportOn="ALL"#settings for when to export training episdoes
    self.bestCrossValError=None
    self.bestTrainingEpisode=-1#index for best training episode
    #settings: BEST(all that beat the best final crossval error)  ALL
    self.crossValRegressionData=[]

  #can use one or both
  #TODO set minimIterations
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

  def runQue(self,saveDirectory=".\\runs"):
    for function in self.trainingQue:
      #Garbage collector should be deleteing these once we're done with them
      self.currentTrainingEpisode=function()
      finalCrossValError = self.runEpisode()

      #TEMP 
      print(self.currentTrainingEpisode.crossValRegressionHistory[-1])
      print(self.currentTrainingEpisode.crossValRegressionVariables)

      bestTrainingEpisode=False

      try:
        mkdir(saveDirectory)
      except FileExistsError as e:
        pass
      except Exception as e:
        print(e)

      if self.bestCrossValError!=None:
        if finalCrossValError<self.bestCrossValError:
          self.bestCrossValError=finalCrossValError
          bestTrainingEpisode=True
      else:
        self.bestCrossValError=finalCrossValError
        bestTrainingEpisode=True

      if self.exportOn=="BEST":
        if bestTrainingEpisode:
          print("saving network to:",saveDirectory)
          self.currentTrainingEpisode.exportNetwork(saveDirectory)
      elif self.exportOn=="ALL":
        print("saving network to:",saveDirectory)
        self.currentTrainingEpisode.exportNetwork(saveDirectory)
      #export data
      
      self.currentTrainingEpisode.exportData(saveDirectory)

  def crossValCallback(self,iteration,crossValError):
    print("\t"+str(crossValError),end="")
    self.lastCrossVal=crossValError
    self.lastCrossValDerivativeEstimation=self.currentTrainingEpisode.crossValDerivativeEstimation(iteration)
    print("\t"+str(self.lastCrossValDerivativeEstimation),end="") 

  def crossValRegressionCallback(self,iteration,crossValRegressionError,crossValRegressionVariables):
    #log data
    self.crossValRegressionData.append([iteration,crossValRegressionError,crossValRegressionVariables])

  def runEpisode(self):
    print("Running episode",self.currentTrainingEpisode.name)
    print()
    print("I\ttrainingError\titerationTime",end="")
    print("\tcrossValError\tcrossValDerivativeEstimation",end="")
    print()
    #training variables
    self.lastCrossValDerivativeEstimation=-1
    self.lastCrossVal=-1

    running=True
    #setupCallbacks
    iterationCallback=lambda iteration,trainingError,iterationTime: print(str(iteration)+"\t"+str(trainingError)+"\t"+str(iterationTime),end="")
    while running:
      self.currentTrainingEpisode.train(iterationCallback,self.crossValCallback,self.crossValRegressionCallback)
      print()
      #check exit requirements
      if self.maxIterationsConstraint:
        if self.currentTrainingEpisode.iterationCounter>=self.maxIterations:
          print("MAX iterations hit, exiting")
          running=False
      elif self.minErrorDerivativeConstraint:
        if self.lastCrossValDerivativeEstimation!=-1 and self.lastCrossValDerivativeEstimation<self.minErrorDerivative:
          print("MIN error derrivate hit, exiting")
          running=False
    return self.lastCrossVal