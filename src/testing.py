#A  769   daisy
#B  1052  dandylion
#C  784   rose
#D  734   sunflower
#E  984   tulip

#%%
import Elinvar
from PIL import Image
import tensorflow as tf
import numpy as np
import random
import time
#%%
learningRate=1e-8

trainingIterations=15000
batchSize=200

shuffleSeed=4739375

crossValSetSize=200
crossValSetSeed=48964
interationsPerCrossVal=1

layerMakeup=[]
runName="B6"

upload=True
#%%
myNet=Elinvar.NN.Network(256,3,debug=False)

input1 = Elinvar.NN.Nodes.InputNode()
input1.setup([256,256,3])

conv1= Elinvar.NN.ConvolutionLayer()
conv1.newLayer(11,48,3,0)
conv1.connect([input1])

pool1=Elinvar.NN.Nodes.MaxPoolingNode()
pool1.newLayer(5,2)
pool1.connect([conv1])

conv2=Elinvar.NN.ConvolutionLayer()
conv2.newLayer(11,48,3,0)
conv2.connect([pool1])

pool2 = Elinvar.NN.MaxPoolingNode()
pool2.newLayer(3,1)
pool2.connect([conv2])

flatten1=Elinvar.NN.Nodes.FlattenNode()
flatten1.connect([pool2])

dense1=Elinvar.NN.Nodes.DenseLayer()
dense1.newLayer(128,"relu")
dense1.connect([flatten1])

dense2=Elinvar.NN.Nodes.DenseLayer()
dense2.newLayer(64,"relu")
dense2.connect([dense1])

dense3=Elinvar.NN.Nodes.DenseLayer()
dense3.newLayer(5,"sigmiod")

myNet.addInputNodes([input1])
myNet.addNodes([conv1,pool1,conv2,pool2,flatten1,dense1,dense2])
myNet.addOutputNodes([dense3])

myNet.build()

totalTrainableVariables=myNet.getTotalTrainableVariables()
ttvText=str(totalTrainableVariables)
for i in range(len(ttvText)):
  print(ttvText[i],end="")
  if((len(ttvText)-i-1)%3==0 and i!=len(ttvText)-1):
    print(",",end="")
print(" trainable variables")
#%%
#w and b here
if upload:
  import wandb
  wandb.init(config={
    "learning rate":learningRate,
    "number of layers":len(layerMakeup),
    "layer makeup":str(layerMakeup),
    "total trainable variables":totalTrainableVariables,
    "batchSize":batchSize,
    "crossValSetSize":crossValSetSize,
    "crossValSetSeed":crossValSetSeed
    },
  project="flowerRecognition",
  entity='japaneserhino')
#%%
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
#%%
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
#deterministically get crossvalset
preShuffledFiles=[i for i in files]
crossValSet=[]
random.seed(crossValSetSeed)
for i in range(crossValSetSize):
  x=random.randint(0,len(preShuffledFiles)-1 )
  crossValSet.append(preShuffledFiles[x])
  del preShuffledFiles[x]

crossValX,crossValY=getBatch(crossValSet)
#%%
#determanistic shuffle
shuffledFiles=[]
random.seed(shuffleSeed)
while len(preShuffledFiles)!=0:
  x=random.randint(0,len(preShuffledFiles)-1)
  shuffledFiles.append(preShuffledFiles[x])
  del preShuffledFiles[x]

dataSize=len(shuffledFiles)
print("Data Size:",dataSize)
#%%
batchNumber=0

for i in range(trainingIterations): 
  startTime=time.time()
  #getBatch
  start=batchSize*batchNumber
  start%=dataSize
  end=start+batchSize
  if(end>=dataSize):
    end%=dataSize
    x,y=getBatch(shuffledFiles[start:]+shuffledFiles[:end])
  else:
    x,y=getBatch(shuffledFiles[start:end])
  batchNumber+=1

  #train
  trainingError= myCNN.train(x,y,learningRate,0)
  crossValError=myCNN.validate(crossValX,crossValY)
  iterationTime=time.time()-startTime
  startTime=time.time()
  #todo holdout error
  if upload:
    wandb.log({
      "index":i,
      "training error":trainingError,
      "cross validation error":crossValError,
      "iterationTime:":iterationTime})
  print(str(i)+"\ttraining error\t"+str(float(trainingError)),end="")
  print("\tcrossval error\t"+str(float(crossValError)),end="")
  print("\titerationtime\t"+str(iterationTime),end="")
  print()
print("FINAL CROSS VAL ERROR\t"+str(float(crossValError)))
if upload:
  wandb.log({"finalTrainingError":crossValError})


# %%
#export the net
print("Exporting...")
myCNN.exportNetwork("./"+runName)
