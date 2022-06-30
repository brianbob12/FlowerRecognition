import numpy as np
import tensorflow as tf
from PIL import Image

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
    x.append(np.asfarray(im))
  #endfor
  return tf.constant(np.asfarray(x),dtype=tf.float32),np.asfarray(y)

#get the data
def getFiles():
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
  return files