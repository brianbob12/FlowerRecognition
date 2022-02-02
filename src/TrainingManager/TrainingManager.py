from random import triangular
from typing import final

from numpy import cross
import TrainingEpisode

#a class to manage training using training episodes

class TrainingManager: 
  def __init__(self):
    self.datasets={}
    self.trainingQue=[]#list of labbda functions to create trainng episodes
    self.currentTrainingEpisode=None
    self.exportOn="ALL"#settings for when to export training episdoes
    self.bestCrossValError=None
    self.bestTrainingEpisode=-1#index for best training episode
    #settings: BEST(all that beat the best final crossval error)  ALL
    self.crossValRegressionData=[]

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

  def runQue(self):
    for function in self.trainingQue:
      self.currentTrainingEpisode=function()
      finalCrossValError = self.runEpisode()

      bestTrainingEpisode=False
      if self.bestCrossValError!=None:
        if finalCrossValError<self.bestCrossValError:
          self.bestCrossValError=finalCrossValError
          bestTrainingEpisode=True
      else:
        self.bestCrossValError=finalCrossValError
        bestTrainingEpisode=True

      if self.exportOn=="BEST":
        if bestTrainingEpisode:
          self.currentTrainingEpisode.exportNetwork()
      elif self.exportOn=="ALL":
        self.currentTrainingEpisode.exportNetwork()

  def crossValCallback(self,iteration,crossValError):
    print("\t"+str(crossValError),end="")
    self.lastCrossVal=crossValError
    self.lastCrossValDerivativeEstimation=self.currentTrainingEpisode.crossValDerivativeEstimation(iteration)
    print("\t"+str(self.lastCrossValDerivativeEstimation),end="") 

  #TODO figure out what to do here
  def crossValRegressionCallback(self,iteration,crossValRegressionError,crossValRegressionVariables):
    #log data
    self.crossValRegressionData.append([iteration,crossValRegressionError,crossValRegressionVariables])

  def runEpisode(self):
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