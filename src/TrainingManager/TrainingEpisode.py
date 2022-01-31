import math
import random
import time
#a class to manage one training run

class TrainingEpisode:
  def __init__(self):
    self.useWandB=False
    self.crossValErrorHistory=[]
    self.trainingErrorHistory=[]
    self.iterationCounter=0
    pass

  def instantiateLearningConfig(self,learningRate,batchSize):
    self.learningRate=learningRate
    self.batchSize=batchSize

  def instantiateMonitoringConfig(self,
  iterationsPerCrossValSample,
  iterationsPerCrossValRegress,
  trainingErrorRegression,
  crossValRegressionLR,
  crossValRegressionSeed):
    self.iterationsPerCrossValSample=iterationsPerCrossValSample
    self.iterationsPerCrossValRegress=iterationsPerCrossValRegress
    self.trainingErrorRegression=trainingErrorRegression
    self.crossValRegressionLR=crossValRegressionLR

    random.seed(crossValRegressionSeed)
    self.crossValRegressionVariables={
      "A":random.random(),
      "B":random.random(),
      "C":random.random(),
      "D":random.random()
    }

  #takes a setup CNN
  def importNetwork(self,CNN):
    self.CNN=CNN
    #setup layerMakeup list for recrods
    self.layerMakeup=[] 
    for i,layer in enumerate(CNN.layers):
      layerType=CNN.layerKey[i]
      if(layerType=="CONVOLUTE"):
        tad=layerType
        tad+="-"+layer.numberOfKernels
        tad+="-"+layer.kernelSize
        tad+="-"+layer.stride
      elif(layerType=="TRANSPOSECONVOLUTE"):
        tad=layerType
        tad+="-"+layer.numberOfKernels
        tad+="-"+layer.kernelSize
        tad+="-"+layer.stride
      elif(layerType=="POOL"):
        tad=layerType
        tad+="-"+layer.size
        tad+="-"+layer.stride
      elif(layer=="FLATTEN"):
        tad=layerType
      elif(layer=="DENSE"):
        tad=layerType
        tad+="-"+layer.size
        tad+="-"+layer.activationKey
      elif(layer=="INSTANCENORMALIZATION"):
        tad=layerType
        tad+="-"+layer.mean
        tad+="-"+layer.stddeve
      elif(layer=="ADAIN"):
        #adaptive instance normilization
        #TODO
        pass
      self.layerMakeup.append(tad)
 
    self.totalTrainableVariables=CNN.totalTrainableVariables()

  #takes a dict: {files,extract}
  #extract is a function returning x,y given a list of files
  def setDataSet(self,dataset,datasetShuffleSeed,crossValSetSize,crossValSetSelectionSeed):
    self.trainingDataSize=len(dataset["files"])-crossValSetSize
    self.crossValSize=crossValSetSize
    self.crossValSelectionSeed=crossValSetSelectionSeed

    self.dataset={
      "trainingFiles":[],
      "crossValFiles":[],
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
    self.dataset["crossValFiles"]=crossValSet

    #deterministic shuffle
    shuffledFiles=[]   
    random.seed(datasetShuffleSeed)
    while len(preShuffledFiles)!=0:
      x=random.randint(0,len(preShuffledFiles)-1)
      shuffledFiles.append(preShuffledFiles[x])
      del preShuffledFiles[x]
    
    self.dataset["trainingFiles"]=shuffledFiles

  def setUpWandB(self):
    self.useWandB=True
    import wandb
    wandb.init(config={
      "learningRate":self.learningRate,
      "numberOfLayers":len(self.layerMakeup),
      "layer makeup":self.layerMakeup,
      "total trainable variables":self.totalTrainableVariables,
      "batchSize":self.batchSize,
      "crossValSetSize":self.crossValSetSize,
      "crossValSeed":self.crossValSelectionSeed
    })

  #training functions

  def crossValEstimation(self,iterationCounter):
    value=self.crossValRegressionVariables["A"]*math.exp(self.crossValRegressionVariables["B"]*iterationCounter)
    value+=self.crossValRegressionVariables["C"]*iterationCounter
    value+=self.crossValRegressionVariables["D"]
    return value

  def crossValDerivativeEstimation(self,iterationCounter):
    dvdi=self.crossValRegressionVariables["A"]*self.crossValRegressionVariables["B"]+math.exp(self.crossValRegressionVariables["B"]*iterationCounter)
    dvdi+=self.crossValRegressionVariables["C"]
    return dvdi

  #crossValMeasurement must be float
  def crossValRegression(self,crossValMeasurements,iterationCounters):
    deda=0
    dedb=0
    dedc=0
    dedd=0
    n=len(crossValMeasurements)
    error=0
    for i in range(crossValMeasurements):
      crossValEstimation=self.crossValDerivativeEstimation(iterationCounters[i])
      error+=(crossValEstimation-crossValMeasurements[i])**2
      pdedv=-2*crossValMeasurements[i]-2*crossValEstimation

      #A
      pdeda=math.exp(self.crossValRegressionVariables["B"]*iterationCounters[i])
      pdeda*=pdedv
      #B
      pdedb=self.crossValRegressionVariables["A"]*iterationCounters[i]*math.exp(self.crossValRegressionVariables["B"]*iterationCounters[i])
      pdedb*=pdedv
      #C
      pdedc=iterationCounters[i]*pdedv 
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
    error=n

    #update perameters
    self.crossValRegressionVariables["A"]-=self.crossValRegressionLR*deda
    self.crossValRegressionVariables["B"]-=self.crossValRegressionLR*dedb
    self.crossValRegressionVariables["C"]-=self.crossValRegressionLR*dedc  
    self.crossValRegressionVariables["D"]-=self.crossValRegressionLR*dedd
    
    return error

  def train(self):
    #getBatch
    start=self.batchSize*self.iterationCounter
    start%=self.trainingDataSize
    end=start+self.batchSize
    if(end>=self.dataSize):
      end%=self.dataSize
      x,y=self.dataset["extract"](self.dataset["training"][start:]+self.dataset["training"][:end])
    else:
      x,y=self.dataset["extract"](self.dataset["training"][start:end])

    #start training
    startTime=time.time()
    trainingError=self.CNN.train(x,y,self.learningRate,0)
    iterationTime=time.time()-startTime

    if(self.iterationCounter%self.iterationsPerCrossValSample==0):
      

    self.iterationCounter+=1


  