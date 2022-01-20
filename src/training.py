#A  769   daisy
#B  1052  dandylion
#C  784   rose
#D  734   sunflower
#E  984   tulip

#%%
from CyrusCNN.CNN import CNN as CNN
from PIL import Image
import tensorflow as tf
import numpy as np
import random
#%%
learningRate=1e-5
trainingIterations=10
batchSize=100
shuffleSeed=4739375
crossValSetSize=200
crossValSetSeed=48964
layerMakeup=[]
runName="A0"
#%%
myCNN=CNN(256,False)

myCNN.addConvolutionLayer(3,1)
layerMakeup.append("CONV-3-1")
myCNN.addPoolingLayer(3,1)
layerMakeup.append("POOL-3-1")
myCNN.addFlattenLayer()
layerMakeup.append("FLATTEN")
myCNN.addDenseLayer(10,"relu")
layerMakeup.append("DENSE-10-relu")
myCNN.addDenseLayer(5,"relu")
layerMakeup.append("DENSE-5-relu")

totalTrainableVariables=myCNN.totalTrainableVariables
#%%
#w and b here
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
  x=random.randint(0,len(preShuffledFiles))
  crossValSet.append(preShuffledFiles[x])
  del preShuffledFiles[x]

crossValX,crossValY=getBatch(crossValSet)
#%%
#determanistic shuffle

random.seed(shuffleSeed)
while len(preShuffledFiles)!=0:
  x=random.randint(0,len(preShuffledFiles))
  shuffledFiles.append(preShuffledFiles[x])
  del preShuffledFiles[x]
shuffledFiles=[]

dataSize=len(shuffledFiles)
#%%
batchNumber=0

for i in range(trainingIterations): 
  #getBatch
  start=batchSize*batchNumber
  start%=dataSize
  end=start+batchSize
  if(end>=dataSize):
    end%=dataSize
    x,y=getBatch(shuffledFiles[start:])
    nx,ny=getBatch(shuffledFiles[:end])
    x+=nx
    y+=ny
  else:
    x,y=getBatch(shuffledFiles[start:end])
  batchNumber+=1

  #train
  trainingError= myCNN.train(x,y,learningRate,0)
  crossValError=myCNN.validate(crossValX,crossValY)
  #todo holdout error
  wandb.log({"index":i,
  "training error":trainingError,
  "cross validation error":crossValError})
  print(str(i)+"\ttraining error\t"+str(float(trainingError)),end="")
  print("\tcrossval error\t"+str(float(crossValError)),end="")
  print()
print("FINAL CROSS VAL ERROR\t"+str(float(crossValError)))
wandb.log({"finalTrainingError":crossValError})


# %%
#export the net
print("Exporting...")
myCNN.exportNetwork("./"+runName)
