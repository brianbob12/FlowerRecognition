#%%
import Elinvar
import tensorflow as tf
import numpy as np
from PIL import Image
#%%
#STEP 1
#setup learning manager and dataset
tm=Elinvar.TM.TrainingManager()
tm.setEpisodeEndRequirements(maxIterations=5000)

#outputs x and y arrays
def getBatch(fileNames):
  x=[]
  y=[]
  for file in fileNames:
    if file[0]=="A":
      y.append([1,0,0,0,0])
    elif file[0]=="B":
      y.append([0,1,0,0,0])
    elif file[0]=="C":
      y.append([0,0,1,0,0])
    elif file[0]=="D":
      y.append([0,0,0,1,0])
    elif file[0]=="E":
      y.append([0,0,0,0,1])

    im=Image.open("./data/flowers/"+file)
    x.append(np.asarray(im))
  #endfor
  return tf.constant(np.asarray(x),dtype=tf.float32),np.asarray(y)

#get the data
a=769
b=1052
c=784
d=734
e=984

#for now use all data in batch
n=a+b+c+d+e
files=[]
for i in range(a):
  files.append("A"+str(i)+".jpg") 
for i in range(b):
  files.append("B"+str(i)+".jpg") 
for i in range(c):
  files.append("C"+str(i)+".jpg") 
for i in range(d):
  files.append("D"+str(i)+".jpg") 
for i in range(e):
  files.append("E"+str(i)+".jpg") 

tm.addDataSet("FlowerDataset",files,getBatch)
#%%
#STEP 2
#DEFINE training episodes

def seriesX(name,learningRate):
  te=Elinvar.TM.TrainingEpisode(name)
  te.instantiateLearningConfig(learningRate,100)
  te.instantiateMonitoringConfig(1,5,False,1e-5,crossValRegressionIterationCount=1000)
  te.setDataSet(tm.datasets["FlowerDataset"],4452,200,48964)
  
  #NOTE: it is important to create a new CNN for each series
  #otherwise each training episode will continue with the same CNN
  #(unless that's what you want)
  myErrorFunction=Elinvar.NN.ErrorFunctions.SoftmaxCrossEntropyWithLogits()

  myNet=Elinvar.NN.SequentialNeuralNetwork(256,3,myErrorFunction)
  myNet.addConvolutionLayer(48,11,3)
  myNet.addPoolingLayer(5,2)
  myNet.addConvolutionLayer(48,3,1)
  myNet.addInstanceNormalizationLayer(1,1)
  myNet.addPoolingLayer(3,1)
  myNet.addConvolutionLayer(48,3,1)
  myNet.addInstanceNormalizationLayer(1,1)
  myNet.addPoolingLayer(3,1)
  myNet.addFlattenLayer()
  myNet.addDenseLayer(128,"relu")
  myNet.addDenseLayer(64,"relu")
  myNet.addDenseLayer(5,"sigmoid")

  te.importNetwork(myNet)

  #must be done after network imported
  #te.setUpWandB("flowerRecognition","japaneserhino")
  return te


def trainingEpisodeGenerator():
  for i in range(20):
    yield lambda : (seriesX("r"+str(i),1e-9*(i+1)))

tm.trainingQue=[lambda: seriesX("testA",1e-9)]
#%%
#STEP 3
#start training
tm.runQue()

# %%
