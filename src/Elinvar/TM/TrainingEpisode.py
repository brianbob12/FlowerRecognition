import random
import time
from typing import Any, Callable, Optional

from Elinvar.NN.Network import Network
from .Exceptions import missingTrainingProtocol

from Elinvar.NN.TrainingProtocols.TrainingProtocol import TrainingProtocol

#a class to manage one training run

class TrainingEpisode:
  def __init__(self,name:str,network:Network,trainingProtocol:TrainingProtocol):
    self.name:str=name
    self.iterationCounter:int=0
    self.network:Network=network
    self.trainingProtocol:TrainingProtocol=trainingProtocol

  def instantiateLearningConfig(self,batchSize:int):
    self.batchSize:int=batchSize

  def instantiateMonitoringConfig(self,
    iterationsPerCrossValSample:int
  ):
    self.iterationsPerCrossValSample:int=iterationsPerCrossValSample

  #takes a dict: {files,extract}
  #extract is a function returning x,y given a list of files
  def setDataSet(
    self,
    dataset,
    datasetShuffleSeed,
    crossValSetSize,
    crossValSetSelectionSeed
    ):
    self.trainingDataSize=len(dataset["files"])-crossValSetSize
    self.crossValSize=crossValSetSize
    self.crossValSelectionSeed=crossValSetSelectionSeed

    self.dataset={
      "trainingFiles":[],
      "crossValx":[],
      "crossValy":[],
      "extract":dataset["extract"]#function to produce x,y for a list of files
    }

    #deterministically get crossvalset
    preShuffledFiles=dataset["files"].copy()
    crossValSet=[]
    random.seed(crossValSetSelectionSeed)
    for i in range(crossValSetSize):
      x=random.randint(0,len(preShuffledFiles)-1 )
      crossValSet.append(preShuffledFiles[x])
      del preShuffledFiles[x]
    self.dataset["crossValx"],self.dataset["crossValy"]=dataset["extract"](crossValSet)

    #deterministic shuffle
    shuffledFiles=[]   
    random.seed(datasetShuffleSeed)
    while len(preShuffledFiles)!=0:
      x=random.randint(0,len(preShuffledFiles)-1)
      shuffledFiles.append(preShuffledFiles[x])
      del preShuffledFiles[x]
    
    self.dataset["trainingFiles"]=shuffledFiles


  def train(
    self,
    iterationCallback:Optional[Callable[[int,float,float],Any]]=None,
    crossValCallback:Optional[Callable[[int,float],Any]]=None
    ):
    if self.trainingProtocol==None:
      raise(missingTrainingProtocol())

    #getBatch
    start=self.batchSize*self.iterationCounter
    start%=self.trainingDataSize
    end=start+self.batchSize
    if(end>=self.trainingDataSize):
      end%=self.trainingDataSize
      x,y=self.dataset["extract"](self.dataset["trainingFiles"][start:]+self.dataset["trainingFiles"][:end])
    else:
      x,y=self.dataset["extract"](self.dataset["trainingFiles"][start:end])

    #start training
    startTime=time.perf_counter()
    trainingError=float(self.network.train(x,self.trainingProtocol,[y]))
    #end training
    iterationTime=time.perf_counter()-startTime
    if(iterationCallback!=None):
      iterationCallback(self.iterationCounter,trainingError,iterationTime)

      
    if(self.iterationCounter%self.iterationsPerCrossValSample==0):
      #cross validate
      crossValError=float(self.network.getError(self.dataset["crossValx"],self.trainingProtocol,[self.dataset["crossValy"]]))

      if crossValCallback!=None:
        crossValCallback(self.iterationCounter,crossValError) 

    self.iterationCounter+=1

  def exportNetwork(self,directory):
    #TODO check if already exists to prevent overwritting
    from os import mkdir

    myPath=directory+"\\"+self.name+"\\Model"
    #create directory if one does not already exists
    try:
      mkdir(directory+"\\"+self.name) 
    except FileExistsError:
      pass

    try:
      mkdir(myPath)
    except FileExistsError:
      pass
    self.network.exportNetwork(myPath)
