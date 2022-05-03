#%%
import Elinvar
import tensorflow as tf
import numpy as np
from PIL import Image
#%%
#STEP 1
#setup learning manager and dataset
tm=Elinvar.TM.TrainingManager()
tm.setEpisodeEndRequirements(maxIterations=5)

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
  te.instantiateLearningConfig(100)
  te.instantiateMonitoringConfig(1)
  te.setDataSet(tm.datasets["FlowerDataset"],4452,200,48964)
  
  #NOTE: it is important to create a new CNN for each series
  #otherwise each training episode will continue with the same CNN
  #(unless that's what you want)
  myErrorFunction=Elinvar.NN.ErrorFunctions.SoftmaxCrossEntropyWithLogits()

  myNet=Elinvar.NN.Network()

  input1 = Elinvar.NN.Nodes.InputNode()
  input1.setup([256,256,3])

  conv1= Elinvar.NN.ConvolutionLayer()
  conv1.newLayer(11,48,3,0)
  conv1.connect([input1])

  inst1=Elinvar.NN.InstanceNormalizationNode()
  inst1.newLayer(1,1)
  inst1.connect([conv1])

  pool1=Elinvar.NN.Nodes.MaxPoolingNode()
  pool1.newLayer(5,2)
  pool1.connect([inst1])

  conv2=Elinvar.NN.ConvolutionLayer()
  conv2.newLayer(11,48,3,0)
  conv2.connect([pool1])

  inst2=Elinvar.NN.InstanceNormalizationNode()
  inst2.newLayer(1,1)
  inst2.connect([conv2])

  pool2 = Elinvar.NN.MaxPoolingNode()
  pool2.newLayer(3,1)
  pool2.connect([inst2])

  flatten1=Elinvar.NN.Nodes.FlattenNode()
  flatten1.connect([pool2])

  dense1=Elinvar.NN.Nodes.DenseLayer()
  dense1.newLayer(128,"relu")
  dense1.connect([flatten1])

  dense2=Elinvar.NN.Nodes.DenseLayer()
  dense2.newLayer(64,"relu")
  dense2.connect([dense1])

  dense3=Elinvar.NN.Nodes.DenseLayer()
  dense3.newLayer(5,"sigmoid")
  dense3.connect([dense2])

  myNet.addInputNodes([input1])
  myNet.addNodes([conv1,inst1,pool1,conv2,inst2,pool2,flatten1,dense1,dense2])
  myNet.addOutputNodes([dense3])

  myNet.build()

  te.setNetwork(myNet)

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
