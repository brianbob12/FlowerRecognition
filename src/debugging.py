#%%
import Elinvar
from PIL import Image
import tensorflow as tf
import numpy as np
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
myNet=Elinvar.NN.Network()
input1=Elinvar.NN.Nodes.InputNode()
input1.setup(lambda :getBatch(files[:10]),[256,256,3])
conv1=Elinvar.NN.Nodes.ConvolutionLayer()
conv1.newLayer(5,4,1,0)
conv1.connect([input1])
pool1=Elinvar.NN.Nodes.MaxPoolingNode()
pool1.newLayer(3,1)
pool1.connect([conv1])
flatten1=Elinvar.NN.Nodes.FlattenNode()
flatten1.connect([pool1])
dense1=Elinvar.NN.Nodes.DenseLayer()
dense1.newLayer(2,"linear")
dense1.connect([flatten1])
myNet.addInputNodes([input1])
myNet.addNodes([conv1,pool1,flatten1])
myNet.addOutputNodes([dense1])
#%%
myNet.build(seed=2212)
#%%
#%%
x,y=getBatch(files[:10])
print(myNet.execute({input1.ID:x},[dense1]))
#%%

#%
# %%
