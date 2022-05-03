import math
import random
import time
import Elinvar
import wandb
from .Exceptions import missingTrainingProtocol

from Elinvar.NN.TrainingProtocols.TrainingProtocol import TrainingProtocol

#a class to manage one training run

class TrainingEpisode:
  def __init__(self,name):
    self.name=name
    self.useWandB=False
    self.crossValErrorHistory=[]
    self.trainingErrorHistory=[]
    self.crossValRegressionHistory=[]
    self.iterationCounter=0
    self.network=None
    self.trainingProtocol=None

  def instantiateLearningConfig(self,batchSize):
    self.batchSize=batchSize

  def instantiateMonitoringConfig(self,
  iterationsPerCrossValSample):
    self.iterationsPerCrossValSample=iterationsPerCrossValSample


  def setNetwork(self,network:Elinvar.NN.Network):
    self.network=network 

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

  def setUpWandB(self,project,entity):
    self.useWandB=True 
    wandb.init(config={
      "learningRate":self.learningRate,
      "numberOfLayers":len(self.layerMakeup),
      "layer makeup":self.layerMakeup,
      "total trainable variables":self.totalTrainableVariables,
      "batchSize":self.batchSize,
      "crossValSetSize":self.crossValSize,
      "crossValSeed":self.crossValSelectionSeed
    },
    project=project,
    entity=entity)

  def getDataForUpload(self,trainingError,iterationTime,crossValError):
    data={
      "index":self.iterationCounter,
      "trainingError":trainingError,
      "cross validation error":crossValError,
      "iterationTime":iterationTime,
      "estimatedErrorDerivative":self.crossValDerivativeEstimation(self.iterationCounter)
    }
    return data

  #training functions

  def crossValEstimation(self,iterationCounter):
    value=self.crossValRegressionVariables["A"]*math.exp(self.crossValRegressionVariables["B"]*iterationCounter)
    value+=self.crossValRegressionVariables["C"]*iterationCounter
    value+=self.crossValRegressionVariables["D"]
    return value

  def crossValDerivativeEstimation(self,iterationCounter):
    try:
      dvdi=self.crossValRegressionVariables["A"]*self.crossValRegressionVariables["B"]*math.exp(self.crossValRegressionVariables["B"]*iterationCounter)
      dvdi+=self.crossValRegressionVariables["C"]
      return dvdi
    except Exception as e:
      print(e)
      return None

  #TODO move into it's own class
  #crossValMeasurement must be float
  def crossValRegression(self):
    try:
      deda=0
      dedb=0
      dedc=0
      dedd=0
      n=len(self.crossValErrorHistory)

      if n==0:
        return
      #weight learning rate by trainingExamples
      criticalN=50
      if n<criticalN:
        effectiveLearningRate=n/criticalN * self.crossValRegressionLR
      else:
        effectiveLearningRate=self.crossValRegressionLR

      error=0
      for i in range(n):
        iteration=self.crossValErrorHistory[i]["iteration"]
        recordedError=self.crossValErrorHistory[i]["error"]
        crossValEstimation=self.crossValEstimation(iteration)
        error+=(recordedError-crossValEstimation)**2
        pdedv=-2*recordedError+2*crossValEstimation

        #A
        pdeda=math.exp(self.crossValRegressionVariables["B"]*iteration)
        pdeda*=pdedv
        #B
        pdedb=self.crossValRegressionVariables["A"]*iteration*math.exp(self.crossValRegressionVariables["B"]*iteration)
        pdedb*=pdedv
        #C
        pdedc=iteration*pdedv 
        #D
        pdedd=pdedv

        deda+=pdeda
        dedb+=pdedb
        dedc+=pdedc
        dedd+=pdedd

      deda/=n
      dedb/=n
      dedc/=n
      dedd/=n
      error/=n
      #update perameters
      self.crossValRegressionVariables["A"]-=effectiveLearningRate*deda
      self.crossValRegressionVariables["B"]-=effectiveLearningRate*dedb
      self.crossValRegressionVariables["C"]-=effectiveLearningRate*dedc  
      self.crossValRegressionVariables["D"]-=effectiveLearningRate*dedd
      return error
    except Exception as e:
      print()
      print("CrossVal Regression Failed")
      print(e)
      print("resetting")
      self.crossValRegressionVariables["A"]=0.3
      self.crossValRegressionVariables["B"]=-0.019
      self.crossValRegressionVariables["C"]=-0.0004 
      self.crossValRegressionVariables["D"]=1
      return 0

  #iterationCallback - iteration number, training error,iteration time
  #crossValCallback - iteration number, crossVal error
  #crossValRegression Callback - iterationNumber, crossValRegressionError, crossValregressionVariables
  def train(self,iterationCallback=None,crossValCallback=None,crossValRegressCallback=None,wandbCallback=None):
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

    #store sample
    self.trainingErrorHistory.append({
      "iteration":self.iterationCounter,
      "error":trainingError,
      "time":iterationTime
    })

    if(self.iterationCounter%self.iterationsPerCrossValSample==0):
      #cross validate
      crossValError=float(self.network.getError(self.dataset["crossValx"],self.trainingProtocol,[self.dataset["crossValy"]]))
      #store sample
      self.crossValErrorHistory.append({
        "iteration":self.iterationCounter,
        "error":crossValError
      })
      if wandbCallback!=None:
        wandbCallback(self.getDataForUpload(trainingError,iterationTime,crossValError))

      if crossValCallback!=None:
        crossValCallback(self.iterationCounter,crossValError) 

    self.iterationCounter+=1

  def exportNetwork(self,directory):
    #TODO check if already exists to prevent overwritting
    from os import mkdir

    myPath=directory+"\\Model\\"+self.name
    #create directory if one does not already exists
    try:
      mkdir(directory+"\\Model") 
    except FileExistsError:
      pass
    except Exception as e:
      #TODO make this error
      print(e)
      raise()

    try:
      mkdir(directory+"\\Model\\"+self.name)
    except FileExistsError:
      pass
    except Exception as e:
      print(e)
      raise()
    self.Network.exportNetwork(myPath)

  #exports data
  def exportData(self,directory):
    from os import mkdir
    #create directory if one does not already exists
    accessPath=directory+"\\"+self.name+"\\errorLogs"
    try:
      mkdir(directory+"\\"+self.name)
      mkdir(directory+"\\"+self.name+"\\errorLogs")
    except FileExistsError:
      pass
    except Exception as e:
      #TODO make this error
      raise()
    
    #save error data
    errorData=[["iteration","training error","iteration time","crossValError"]]
    #index=iteration+1
    #iteration, training error, iteration time, crossValError
    for record in self.trainingErrorHistory:
       errorData.append([record["iteration"],record["error"],record["time"]])
    for record in self.crossValErrorHistory:
      errorData[record["iteration"]+1].append(record["error"])
    with open(accessPath+"\\"+"errorData.csv","w") as f:
      for line in errorData:
        l=""
        for var in line:
          l+=str(var)
          l+=","
        f.write(l[:-1])#exclude last comma
        f.write("\n")
    
    #clear some data to save memory
    del errorData
    
    #save crossval regression data
    crossValRegressionData=[["iteration","crossValRegressionError"]+[i for i in self.crossValRegressionVariables.keys()]]
    for record in self.crossValRegressionHistory:
      tad=[record["iteration"],record["error"]]
      for key in self.crossValRegressionVariables.keys():
        tad.append(record[key])
      crossValRegressionData.append(tad)
    with open(accessPath+"//crossValRegressionData.csv","w") as f:
      for line in crossValRegressionData:
        l=""
        for var in line:
          l+=str(var)
          l+=","
        f.write(l[:-1])#exclude last comma
        f.write("\n")
  