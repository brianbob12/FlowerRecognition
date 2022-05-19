#%%
import Elinvar
import tensorflow as tf
import numpy as np
from PIL import Image

#%%
#STEP 1
#setup learning manager and dataset
from Elinvar.TM.Modules import Conditions
tm=Elinvar.TM.TrainingManager()
tm.setEpisodeExitCondition(Conditions.ReachedIteration(90000))
tm.modules+=[
  Elinvar.TM.Modules.Log2Console(),
  Elinvar.TM.Modules.WeightsAndBiases("flowerRecognition","japaneserhino"),
  Elinvar.TM.Modules.Export(exportAll=True)
]

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

#%%
#STEP 2
#DEFINE training episodes

def seriesX(index): 
  name=f"D{index}"
  learningRate=3e-6
  convMove=-5#number of convolution filters chnages by this
  denseMove=10#number of dense neurons changes by this

  #NOTE: it is important to create a new network for each series
  #otherwise each training episode will continue with the same Network
  #(unless that's what you want)

  myNet=Elinvar.NN.Network()

  input1 = Elinvar.NN.Nodes.InputNode()
  input1.setup([256,256,3])

  conv1= Elinvar.NN.ConvolutionLayer()
  conv1.newLayer(11,48+convMove*index,3,0)
  conv1.connect([input1])

  inst1=Elinvar.NN.InstanceNormalizationNode()
  inst1.newLayer(1,1)
  inst1.connect([conv1])

  pool1=Elinvar.NN.Nodes.MaxPoolingNode()
  pool1.newLayer(5,2)
  pool1.connect([inst1])

  conv2=Elinvar.NN.ConvolutionLayer()
  conv2.newLayer(11,48+convMove*index,3,0)
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
  dense1.newLayer(128+denseMove*index,"relu")
  dense1.connect([flatten1])

  dense2=Elinvar.NN.Nodes.DenseLayer()
  dense2.newLayer(64+denseMove*index,"relu")
  dense2.connect([dense1])

  dense3=Elinvar.NN.Nodes.DenseLayer()
  dense3.newLayer(5,"sigmoid")
  dense3.connect([dense2])

  myNet.addInputNodes([input1])
  myNet.addNodes([conv1,inst1,pool1,conv2,inst2,pool2,flatten1,dense1,dense2])
  myNet.addOutputNodes([dense3])

  myNet.build()



  myErrorFunction=Elinvar.NN.ErrorFunctions.SoftmaxCrossEntropyWithLogits()
  myTrainingProtocol=Elinvar.NN.XYTraining(learningRate,tf.keras.optimizers.Adam,[dense3],myErrorFunction)

  te=Elinvar.TM.TrainingEpisode(name,myNet,myTrainingProtocol)
  te.instantiateLearningConfig(200)
  te.instantiateMonitoringConfig(5)

  #deal with dataset
  def extract(files):
    x,y=getBatch(files)
    return {input1.ID:x},y
  myDataset={
    "files":files,
    "extract":extract
  }
  te.setDataSet(myDataset,4452,200,48964)

  return te


trainingQue=[]
for i in range(6):
  trainingQue.append((lambda v:(lambda : seriesX(v)))(i))

tm.trainingQue=trainingQue
#%%
#STEP 3
#start training
tm.runQue()

# %%
